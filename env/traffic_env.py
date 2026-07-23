"""
Gymnasium RL Environment: TrafficEnv
- State (Observation): 2D Grid matrix (8 lanes x 10 cells) + Current phase + Remaining countdown duration.
- Action: 0 = Keep current green phase (advances action_step_delta seconds), 1 = Switch phase (fast-forwards 10s countdown).
- Reward: -(alpha * queue_length + beta * waiting_time).
- Optimizations: libsumo/traci fallback, unique process sim_label, yellow countdown fast-forward, frame skipping.
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

# Ensure TraCI is importable from SUMO_HOME
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)

import traci

class TrafficEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, config_path="config.yaml"):
        super().__init__()
        
        # Load configuration file
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.sumo_cfg = os.path.abspath(self.cfg["env"]["sumo_cfg"])
        self.use_gui = self.cfg["env"]["use_gui"]
        self.use_libsumo = self.cfg["env"].get("use_libsumo", True)
        self.tl_id = self.cfg["env"]["tl_id"]
        self.countdown_duration = self.cfg["env"]["countdown_duration"]
        self.action_step_delta = self.cfg["env"].get("action_step_delta", 3)

        # Automatically recompile the intersection network (.net.xml) from XML config files
        # Only recompile in main process / rank 0 or if intersection.net.xml does not exist yet
        try:
            net_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sumo_config", "intersection.net.xml")
            if not os.path.exists(net_file) or os.environ.get("SUMO_WORKER_RANK", "0") == "0":
                from build_net import build_network
                build_network()
        except Exception:
            pass
        
        self.cell_length = self.cfg["grid"]["cell_length"]
        self.n_cells = self.cfg["grid"]["n_cells"]
        self.type_map = self.cfg["grid"]["vehicle_type_code"]
        
        self.alpha = self.cfg["reward"]["alpha"]
        self.beta = self.cfg["reward"]["beta"]

        # Yellow countdown phases (in tll.tll.xml, phase 1 and phase 3 are yellow)
        self.yellow_phases = {1, 3}
        
        # 8 incoming lanes to intersection J0: [N2J0_0, N2J0_1, S2J0_0, S2J0_1, E2J0_0, E2J0_1, W2J0_0, W2J0_1]
        # Observation Space structure:
        # Flattened grid (8 * 10 = 80) + 1 (Phase ID) + 1 (Countdown remaining) = 82
        obs_dim = (8 * self.n_cells) + 2
        self.observation_space = spaces.Box(
            low=0, high=max(10, self.countdown_duration), shape=(obs_dim,), dtype=np.float32
        )
        
        # Action Space: Discrete(2) -> 0: Keep green, 1: Switch (trigger 10s countdown)
        self.action_space = spaces.Discrete(2)
        
        self.all_lanes = []
        self.current_step = 0
        # Unique label per process & instance to prevent port conflicts in SubprocVecEnv
        self.sim_label = f"sim_{os.getpid()}_{id(self)}"
        self.using_libsumo = False
        self.conn = None

    def _get_sumo_binary(self) -> str:
        import shutil
        binary_name = "sumo-gui" if self.use_gui else "sumo"
        
        # 1. Search in system PATH
        binary_path = shutil.which(binary_name)
        if binary_path:
            return binary_path
            
        # 2. Search via SUMO_HOME environment variable
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            candidate = os.path.join(sumo_home, "bin", f"{binary_name}.exe")
            if os.path.exists(candidate):
                return candidate
                
        # 3. Fallback search in default Windows installation paths
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
                # Automatically set SUMO_HOME if found
                os.environ["SUMO_HOME"] = os.path.dirname(p)
                return candidate

        # Raise detailed guidance error if SUMO executable cannot be found
        raise FileNotFoundError(
            f"\n❌ ERROR: SUMO SIMULATOR ('{binary_name}') NOT FOUND!\n"
            f"👉 SUMO simulation software is not installed or not added to your system PATH.\n\n"
            f"🛠 EASY SUMO INSTALLATION GUIDE:\n"
            f"Method 1: Open PowerShell and run the automatic installer command:\n"
            f"          winget install DLR.SUMO\n\n"
            f"Method 2: Download the official .msi installer from the SUMO website:\n"
            f"          https://sumo.dlr.de/docs/Downloads.php\n\n"
            f"After installation, reopen your terminal and run 'python test_env.py' again!"
        )

    def _kill_old_sumo_processes(self):
        """Kill all stale SUMO processes to prevent TraCI port conflict errors."""
        import subprocess
        for proc_name in ["sumo-gui.exe", "sumo.exe"]:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", proc_name, "/T"],
                    capture_output=True, timeout=3
                )
            except Exception:
                pass
        import time
        time.sleep(0.5)  # Wait briefly for OS to release socket port

    def _init_library(self):
        """Choose between libsumo (fast C++ embedding) and traci (socket IPC)."""
        if not self.use_gui and self.use_libsumo:
            try:
                import libsumo
                self.traci_lib = libsumo
                self.using_libsumo = True
                return
            except (ImportError, Exception):
                self.using_libsumo = False
        self.traci_lib = traci
        self.using_libsumo = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # Close previous connection if open
        self.close()
        self._init_library()

        if not self.using_libsumo:
            # Do NOT kill system-wide sumo.exe processes when running parallel workers inside SubprocVecEnv,
            # because taskkill /F /IM sumo.exe would kill the active sumo instances of all other parallel workers!
            if os.environ.get("IN_VEC_ENV", "0") != "1":
                self._kill_old_sumo_processes()
            sumo_binary = self._get_sumo_binary()
            cmd = [
                sumo_binary,
                "-c", self.sumo_cfg,
                "--no-warnings", "true",
                "--start", "true",
                "--step-length", "1.0"
            ]
            self.traci_lib.start(cmd, label=self.sim_label)
            self.conn = self.traci_lib.getConnection(self.sim_label)
        else:
            cmd = [
                "sumo",
                "-c", self.sumo_cfg,
                "--no-warnings", "true",
                "--step-length", "1.0"
            ]
            self.traci_lib.start(cmd)
            self.conn = self.traci_lib  # libsumo module acts directly as connection
        
        # Retrieve lanes controlled by traffic light J0
        # Filter and sort to guarantee fixed order inside the observation matrix
        raw_controlled = self.conn.trafficlight.getControlledLanes(self.tl_id)
        incoming_prefixes = ("N2J0", "S2J0", "E2J0", "W2J0")
        self.all_lanes = sorted(list(set([l for l in raw_controlled if l.startswith(incoming_prefixes)])))
        
        # Fallback to default 8 incoming lanes if filtering yields unexpected results
        if len(self.all_lanes) != 8:
            self.all_lanes = [
                "N2J0_0", "N2J0_1",
                "S2J0_0", "S2J0_1",
                "E2J0_0", "E2J0_1",
                "W2J0_0", "W2J0_1"
            ]

        self.current_step = 0
        self.conn.simulationStep()
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        current_phase = self.conn.trafficlight.getPhase(self.tl_id)
        
        # --- CRITICAL CONSTRAINT & OPTIMIZATION 1: Countdown Lock + Fast-Forward ---
        # If currently in a yellow phase (Phase 1 or Phase 3), fast forward until yellow finishes.
        if current_phase in self.yellow_phases:
            while current_phase in self.yellow_phases and self.current_step < 3600:
                self.conn.simulationStep()
                self.current_step += 1
                current_phase = self.conn.trafficlight.getPhase(self.tl_id)
        else:
            # Agent makes decision during Green phases (Phase 0 or Phase 2)
            if action == 1:
                # Decision: SWITCH PHASE -> Transition to the next Yellow phase (1 or 3)
                next_phase = (current_phase + 1) % 4
                self.conn.trafficlight.setPhase(self.tl_id, next_phase)
                self.conn.trafficlight.setPhaseDuration(self.tl_id, self.countdown_duration)
                
                # FAST-FORWARD: Advance simulation through the entire 10s yellow countdown duration
                for _ in range(self.countdown_duration):
                    if self.current_step >= 3600:
                        break
                    self.conn.simulationStep()
                    self.current_step += 1
            else:
                # Decision: KEEP PHASE -> Maintain current green phase
                # OPTIMIZATION 4: Action Repeat / Frame Skipping for green phase
                for _ in range(self.action_step_delta):
                    if self.current_step >= 3600:
                        break
                    self.conn.simulationStep()
                    self.current_step += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        
        # Termination condition: terminate only after at least 10 steps (allowing vehicles to spawn)
        # when no vehicles remain expected, or when reaching the maximum duration limit of 3600 steps
        sim_ended = (
            self.current_step >= 10
            and self.conn.simulation.getMinExpectedNumber() <= 0
        )
        terminated = sim_ended or self.current_step >= 3600
        truncated = False
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        grid = self._build_grid() # Shape: (8, 10)
        phase = self.conn.trafficlight.getPhase(self.tl_id)
        
        # Calculate remaining countdown duration in the current phase
        next_switch = self.conn.trafficlight.getNextSwitch(self.tl_id)
        current_time = self.conn.simulation.getTime()
        countdown = max(0.0, min(float(self.countdown_duration), next_switch - current_time))
        
        flat_obs = np.concatenate([
            grid.flatten().astype(np.float32),
            np.array([phase, countdown], dtype=np.float32)
        ])
        return flat_obs

    def _build_grid(self) -> np.ndarray:
        """
        Extract vehicle data into an encoded 2D grid matrix:
        Each incoming lane (8 lanes total) is divided into n_cells (10 cells), each cell cell_length (7m) long.
        Encoding schema: 0=Empty, 1=Motorcycle, 2=Passenger Car, 3=Bus.
        """
        grid = np.zeros((len(self.all_lanes), self.n_cells), dtype=np.int8)
        
        for lane_idx, lane_id in enumerate(self.all_lanes):
            try:
                vehicles = self.conn.lane.getLastStepVehicleIDs(lane_id)
                lane_len = self.conn.lane.getLength(lane_id)
                for veh_id in vehicles:
                    pos = self.conn.vehicle.getLanePosition(veh_id)
                    vtype = self.conn.vehicle.getTypeID(veh_id)
                    
                    # Distance from vehicle to intersection stopline
                    dist_to_stopline = lane_len - pos
                    cell = int(dist_to_stopline / self.cell_length)
                    
                    # Assign only if vehicle resides inside our observation grid range (0 to n_cells-1)
                    if 0 <= cell < self.n_cells:
                        code = self.type_map.get(vtype, 2) # Default to passenger car if type not found
                        grid[lane_idx, cell] = code
            except Exception:
                continue
                
        return grid

    def _compute_reward(self) -> float:
        """
        Redesigned 3-component Reward Function to break saturation at -1.46e6:

        Component 1 — THROUGHPUT BONUS (+):
            Reward for every vehicle that departed/exited the intersection this step.
            This is the KEY positive signal that was missing — it gives the AI direct
            credit assignment for making good switching decisions.

        Component 2 — PRESSURE PENALTY (-):
            Penalizes queue and waiting time, normalized per lane so the reward scale
            stays stable regardless of how many vehicles are in the simulation.
            Dividing waiting_time by 100.0 prevents it from dominating the gradient.

        Component 3 — SWITCH COST (-):
            Small penalty when the phase changes to the next GREEN phase (not yellow).
            This discourages rapid phase oscillation that causes unnecessary yellow delays.
        """
        # --- Component 1: THROUGHPUT BONUS ---
        throughput_weight = self.cfg["reward"].get("throughput_weight", 2.0)
        try:
            vehicles_departed = self.conn.simulation.getDepartedNumber()
        except Exception:
            vehicles_departed = 0
        throughput_bonus = vehicles_departed * throughput_weight

        # --- Component 2: NORMALIZED PRESSURE PENALTY ---
        total_queue = 0.0
        total_wait = 0.0
        for lane_id in self.all_lanes:
            try:
                total_queue += self.conn.lane.getLastStepHaltingNumber(lane_id)
                total_wait += self.conn.lane.getWaitingTime(lane_id)
            except Exception:
                continue
        n_lanes = max(1, len(self.all_lanes))
        pressure_penalty = -(
            self.alpha * (total_queue / n_lanes)
            + self.beta * (total_wait / n_lanes / 100.0)
        )

        # --- Component 3: SWITCH COST ---
        switch_penalty_val = self.cfg["reward"].get("switch_penalty", -2.0)
        try:
            current_phase = self.conn.trafficlight.getPhase(self.tl_id)
        except Exception:
            current_phase = getattr(self, "_last_phase", 0)
        switch_cost = 0.0
        if (hasattr(self, "_last_phase")
                and current_phase != self._last_phase
                and current_phase not in self.yellow_phases):
            switch_cost = switch_penalty_val
        self._last_phase = current_phase

        reward = throughput_bonus + pressure_penalty + switch_cost
        return float(reward)

    def close(self):
        try:
            if hasattr(self, "conn") and self.conn and not self.using_libsumo:
                self.conn.close()
        except Exception:
            pass
        try:
            if hasattr(self, "traci_lib") and hasattr(self.traci_lib, "isLoaded") and self.traci_lib.isLoaded():
                self.traci_lib.close()
        except Exception:
            pass
        self.conn = None
