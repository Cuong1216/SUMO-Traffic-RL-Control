"""
MultiAgentTrafficEnv — Independent Learners MARL Environment for 2x2 Grid Traffic Control
Architecture: 1 SUMO simulation instance, N agents (1 per traffic light).
Each agent observes: local grid (8x10) + current phase (1) + countdown (1) + neighbor phases (2).
"""
import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import yaml
import traci
import gymnasium as gym
from gymnasium import spaces

TL_IDS = ["J00", "J01", "J10", "J11"]  # 2x2 grid traffic light IDs

# Neighbor map: Green Wave coordination signal
NEIGHBORS = {
    "J00": ["J01", "J10"],
    "J01": ["J00", "J11"],
    "J10": ["J00", "J11"],
    "J11": ["J01", "J10"],
}


class SingleTLAgent:
    """
    Represents one traffic-light agent within the shared simulation.
    Holds per-agent observation/action spaces and local state.
    """
    def __init__(self, tl_id: str, cfg: dict):
        self.tl_id = tl_id
        self.cfg = cfg
        self.n_cells = cfg["grid"]["n_cells"]
        self.cell_length = cfg["grid"]["cell_length"]
        self.type_map = cfg["grid"]["vehicle_type_code"]
        self.yellow_phases = {1, 3}
        self.countdown_duration = cfg["env"]["countdown_duration"]
        self.action_step_delta = cfg["env"].get("action_step_delta", 3)
        self._last_phase = 0
        self.all_lanes: list[str] = []

        n_neighbors = len(NEIGHBORS[tl_id])  # 2 neighbors per junction in 2x2 grid
        # obs = local_grid(8 * n_cells) + phase(1) + countdown(1) + neighbor_phases(2)
        obs_dim = (8 * self.n_cells) + 2 + n_neighbors
        self.observation_space = spaces.Box(
            low=0, high=max(10, self.countdown_duration),
            shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

    def get_obs(self, conn) -> np.ndarray:
        grid = self._build_grid(conn)
        phase = conn.trafficlight.getPhase(self.tl_id)
        next_switch = conn.trafficlight.getNextSwitch(self.tl_id)
        current_time = conn.simulation.getTime()
        countdown = max(0.0, min(float(self.countdown_duration), next_switch - current_time))

        # Cooperative signal: observe neighbors' current phase
        neighbor_phases = [
            float(conn.trafficlight.getPhase(nb)) for nb in NEIGHBORS[self.tl_id]
        ]
        return np.concatenate([
            grid.flatten().astype(np.float32),
            np.array([phase, countdown], dtype=np.float32),
            np.array(neighbor_phases, dtype=np.float32),
        ])

    def compute_reward(self, conn) -> float:
        """
        Hybrid reward:
          - Local pressure penalty at THIS junction (queue & wait time)
          - Global throughput bonus (vehicles departing/exiting simulation)
          - Phase switch cost to discourage rapid oscillation
        """
        cfg = self.cfg
        alpha = cfg["reward"]["alpha"]
        beta = cfg["reward"]["beta"]
        throughput_weight = cfg["reward"].get("throughput_weight", 2.0)
        switch_penalty_val = cfg["reward"].get("switch_penalty", -2.0)
        local_weight = cfg["reward"].get("local_weight", 0.7)
        global_weight = cfg["reward"].get("global_weight", 0.3)

        # Local pressure
        total_queue, total_wait = 0.0, 0.0
        for lane_id in self.all_lanes:
            try:
                total_queue += conn.lane.getLastStepHaltingNumber(lane_id)
                total_wait += conn.lane.getWaitingTime(lane_id)
            except Exception:
                continue
        n = max(1, len(self.all_lanes))
        local_penalty = -(alpha * (total_queue / n) + beta * (total_wait / n / 100.0))

        # Global throughput (shared across all agents in this step)
        try:
            departed = conn.simulation.getDepartedNumber()
        except Exception:
            departed = 0
        global_bonus = departed * throughput_weight

        # Switch cost
        try:
            current_phase = conn.trafficlight.getPhase(self.tl_id)
        except Exception:
            current_phase = self._last_phase
        switch_cost = 0.0
        if (self._last_phase != current_phase and current_phase not in self.yellow_phases):
            switch_cost = switch_penalty_val
        self._last_phase = current_phase

        return float(local_weight * local_penalty + global_weight * global_bonus + switch_cost)

    def _build_grid(self, conn) -> np.ndarray:
        grid = np.zeros((8, self.n_cells), dtype=np.int8)
        for lane_idx, lane_id in enumerate(self.all_lanes[:8]):
            try:
                vehicles = conn.lane.getLastStepVehicleIDs(lane_id)
                lane_len = conn.lane.getLength(lane_id)
                for veh_id in vehicles:
                    pos = conn.vehicle.getLanePosition(veh_id)
                    vtype = conn.vehicle.getTypeID(veh_id)
                    dist = lane_len - pos
                    cell = int(dist / self.cell_length)
                    if 0 <= cell < self.n_cells:
                        grid[lane_idx, cell] = self.type_map.get(vtype, 2)
            except Exception:
                continue
        return grid


class MultiAgentTrafficEnv:
    """
    Wraps N SingleTLAgents sharing ONE SUMO simulation.
    API: reset() -> dict[tl_id -> obs]
         step(actions: dict) -> (obs, rewards, dones, infos)
    """
    def __init__(self, config_path="config_marl.yaml"):
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config_marl.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
            
        self.sumo_cfg = os.path.abspath(self.cfg["env"]["sumo_cfg"])
        self.use_gui = self.cfg["env"].get("use_gui", False)
        self.sim_label = f"grid_{os.getpid()}_{id(self)}"
        self.conn = None
        self.current_step = 0
        self.max_steps = self.cfg["env"].get("max_steps", 3600)
        self.action_step_delta = self.cfg["env"].get("action_step_delta", 3)
        self.countdown_duration = self.cfg["env"]["countdown_duration"]

        # Instantiate one agent per traffic light
        self.agents: dict[str, SingleTLAgent] = {
            tl_id: SingleTLAgent(tl_id, self.cfg) for tl_id in TL_IDS
        }
        self.possible_agents = list(TL_IDS)

    def reset(self, seed=None):
        self._close_sumo()
        self._start_sumo()
        self.current_step = 0

        # Discover controlled lanes per junction in sorted order
        for tl_id, agent in self.agents.items():
            raw = self.conn.trafficlight.getControlledLanes(tl_id)
            lanes = sorted(list(set(raw)))
            agent.all_lanes = lanes[:8] if lanes else list(raw[:8])
            agent._last_phase = self.conn.trafficlight.getPhase(tl_id)

        self.conn.simulationStep()
        obs = {tl_id: agent.get_obs(self.conn) for tl_id, agent in self.agents.items()}
        return obs, {}

    def step(self, actions: dict[str, int]):
        """
        actions: {tl_id: 0 | 1} -> Synchronized step for all agents.
        """
        any_yellow_triggered = False

        # 1. Apply phase switch requests during green phases
        for tl_id, action in actions.items():
            agent = self.agents[tl_id]
            current_phase = self.conn.trafficlight.getPhase(tl_id)
            if current_phase not in agent.yellow_phases and action == 1:
                next_phase = (current_phase + 1) % 4
                self.conn.trafficlight.setPhase(tl_id, next_phase)
                self.conn.trafficlight.setPhaseDuration(tl_id, self.countdown_duration)
                any_yellow_triggered = True
            elif current_phase in agent.yellow_phases:
                any_yellow_triggered = True

        # 2. Advance simulation steps
        # If any junction triggered/is in yellow countdown, step forward until all yellow transitions finish
        if any_yellow_triggered:
            steps_to_run = self.countdown_duration
        else:
            steps_to_run = self.action_step_delta

        for _ in range(steps_to_run):
            if self.current_step >= self.max_steps:
                break
            self.conn.simulationStep()
            self.current_step += 1
            
            # Check if all agents finished their yellow transitions early
            all_green = all(
                self.conn.trafficlight.getPhase(tid) not in self.agents[tid].yellow_phases
                for tid in TL_IDS
            )
            if any_yellow_triggered and all_green:
                break

        obs = {tl_id: agent.get_obs(self.conn) for tl_id, agent in self.agents.items()}
        rewards = {tl_id: agent.compute_reward(self.conn) for tl_id, agent in self.agents.items()}

        sim_ended = (
            self.current_step >= 10
            and self.conn.simulation.getMinExpectedNumber() <= 0
        )
        terminated = sim_ended or self.current_step >= self.max_steps
        dones = {tl_id: terminated for tl_id in TL_IDS}
        dones["__all__"] = terminated

        return obs, rewards, dones, {}

    def get_metrics(self) -> dict:
        """Export real-time metrics for Dashboard."""
        metrics = {}
        for tl_id, agent in self.agents.items():
            total_queue, total_wait = 0.0, 0.0
            for lane_id in agent.all_lanes:
                try:
                    total_queue += self.conn.lane.getLastStepHaltingNumber(lane_id)
                    total_wait += self.conn.lane.getWaitingTime(lane_id)
                except Exception:
                    pass
            metrics[tl_id] = {
                "phase": self.conn.trafficlight.getPhase(tl_id),
                "queue": total_queue,
                "waiting_time": total_wait,
                "step": self.current_step,
            }
        return metrics

    def _start_sumo(self):
        sumo_binary = self._get_sumo_binary()
        cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--no-warnings", "true",
            "--start", "true",
            "--step-length", "1.0"
        ]
        traci.start(cmd, label=self.sim_label)
        self.conn = traci.getConnection(self.sim_label)

    def _close_sumo(self):
        try:
            if self.conn:
                self.conn.close()
        except Exception:
            pass
        try:
            if traci.isLoaded() and self.sim_label in traci.getConnectionLabelList():
                traci.switch(self.sim_label)
                traci.close()
        except Exception:
            pass
        self.conn = None

    def _get_sumo_binary(self) -> str:
        import shutil
        binary_name = "sumo-gui" if self.use_gui else "sumo"
        b = shutil.which(binary_name)
        if b:
            return b
        sumo_home = os.environ.get("SUMO_HOME", "")
        if sumo_home:
            candidate = os.path.join(sumo_home, "bin", f"{binary_name}.exe")
            if os.path.exists(candidate):
                return candidate
        default_paths = [
            r"D:\Eclipse\Sumo\bin",
            r"C:\Program Files (x86)\Eclipse\Sumo\bin",
            r"C:\Program Files\Eclipse\Sumo\bin",
            r"C:\Program Files (x86)\DLR\Sumo\bin",
            r"C:\Program Files\DLR\Sumo\bin",
            r"C:\Sumo\bin"
        ]
        for p in default_paths:
            candidate = os.path.join(p, f"{binary_name}.exe")
            if os.path.exists(candidate):
                os.environ["SUMO_HOME"] = os.path.dirname(p)
                return candidate
        raise FileNotFoundError(f"SUMO simulator ('{binary_name}') not found. Please install or set SUMO_HOME.")

    def close(self):
        self._close_sumo()
