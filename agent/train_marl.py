"""
MARL Training Script: Independent Learners Pattern for 2x2 Grid Traffic Control
- 1 model per traffic light junction (J00, J01, J10, J11 -> 4 models total)
- Each model uses Stable-Baselines3 DQN with a custom Gymnasium-compatible wrapper
- Coordinator loop steps all agents synchronously on shared MultiAgentTrafficEnv
- Exports real-time queue length & waiting time to logs/marl_metrics.json for Streamlit dashboard
"""
import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import json
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.multi_agent_env import MultiAgentTrafficEnv, TL_IDS


class SingleAgentWrapper(gym.Env):
    """
    Wraps MultiAgentTrafficEnv for one specific traffic light (tl_id) agent.
    Provides standard Gymnasium space interfaces for SB3 initialization.
    """
    def __init__(self, marl_env: MultiAgentTrafficEnv, tl_id: str):
        super().__init__()
        self.env = marl_env
        self.tl_id = tl_id
        self.observation_space = marl_env.agents[tl_id].observation_space
        self.action_space = marl_env.agents[tl_id].action_space
        self._last_others_actions = {}

    def reset(self, *, seed=None, options=None):
        obs_dict, info = self.env.reset(seed=seed)
        return obs_dict[self.tl_id], {}

    def step(self, action):
        joint = {tid: self._last_others_actions.get(tid, 0) for tid in self.env.possible_agents}
        joint[self.tl_id] = int(action)
        obs_dict, reward_dict, done_dict, info = self.env.step(joint)
        obs = obs_dict[self.tl_id]
        reward = reward_dict[self.tl_id]
        terminated = done_dict["__all__"]
        return obs, reward, terminated, False, {}


def train_marl(config_path="config_marl.yaml"):
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config_marl.yaml")
        
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("🚀 Initializing 2x2 Grid MARL Environment for Independent Learners...")
    marl_env = MultiAgentTrafficEnv(config_path)
    
    # Pre-compile grid network once right before starting training
    try:
        from build_net_grid import build_grid_network
        build_grid_network()
    except Exception:
        pass

    # Kill any stale SUMO processes before spawning new instance
    try:
        import subprocess
        for proc_name in ["sumo-gui.exe", "sumo.exe"]:
            subprocess.run(["taskkill", "/F", "/IM", proc_name, "/T"], capture_output=True, timeout=3)
    except Exception:
        pass

    model_dir = cfg["agent"]["model_save_dir"]
    log_dir = cfg["agent"]["log_dir"]
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize 4 independent DQN models (one per traffic light)
    models = {}
    for tl_id in TL_IDS:
        wrapper = SingleAgentWrapper(marl_env, tl_id)
        env = DummyVecEnv([lambda w=wrapper: w])
        model_path = os.path.join(model_dir, f"dqn_{tl_id}.zip")
        if os.path.exists(model_path):
            print(f"📦 Resuming agent '{tl_id}' from checkpoint: {model_path}")
            models[tl_id] = DQN.load(model_path, env=env, device="cpu", tensorboard_log=os.path.join(log_dir, tl_id))
            models[tl_id].exploration_rate = 0.1
        else:
            print(f"🧠 Initializing new DQN agent '{tl_id}' from scratch...")
            models[tl_id] = DQN(
                policy="MlpPolicy",
                env=env,
                learning_rate=cfg["agent"]["learning_rate"],
                buffer_size=cfg["agent"].get("buffer_size", 20000),
                batch_size=cfg["agent"]["batch_size"],
                gamma=cfg["agent"]["gamma"],
                tensorboard_log=os.path.join(log_dir, tl_id),
                verbose=0,
                device="cpu"
            )

    total_timesteps = cfg["agent"]["total_timesteps"]
    obs_dict, _ = marl_env.reset()
    step = 0
    episode_rewards = {tid: 0.0 for tid in TL_IDS}
    metrics_file = os.path.join("logs", "marl_metrics.json")
    os.makedirs("logs", exist_ok=True)

    print(f"\n▶️ Starting coordinated MARL training loop across {total_timesteps} steps...")
    
    while step < total_timesteps:
        # 1. Each agent predicts action based on current state & cooperative neighbor phase observation
        actions = {}
        for tl_id in TL_IDS:
            action, _ = models[tl_id].predict(obs_dict[tl_id], deterministic=False)
            actions[tl_id] = int(action)

        # 2. Step multi-agent environment synchronously
        next_obs_dict, reward_dict, done_dict, _ = marl_env.step(actions)
        step += 1

        for tl_id in TL_IDS:
            episode_rewards[tl_id] += reward_dict[tl_id]

        # 3. Store transition in each agent's independent replay buffer and trigger gradient update
        for tl_id in TL_IDS:
            model = models[tl_id]
            model.replay_buffer.add(
                obs=obs_dict[tl_id].reshape(1, -1),
                next_obs=next_obs_dict[tl_id].reshape(1, -1),
                action=np.array([[actions[tl_id]]]),
                reward=np.array([[reward_dict[tl_id]]]),
                done=np.array([[done_dict["__all__"]]]),
                infos=[{}]
            )
            
            # Synchronize SB3 internal step counters for exploration scheduling & logging
            model.num_timesteps = step
            model._current_progress_remaining = max(0.0, 1.0 - (step / float(total_timesteps)))
            model._update_current_progress_remaining(model.num_timesteps, total_timesteps)
            
            if step > model.learning_starts and step % model.train_freq.frequency == 0:
                model.train(batch_size=model.batch_size, gradient_steps=1)

        # 4. Export real-time metrics every 100 steps for Streamlit dashboard
        if step % 100 == 0:
            metrics = marl_env.get_metrics()
            metrics["__meta__"] = {"step": step, "total_timesteps": total_timesteps}
            try:
                with open(metrics_file, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                with open(os.path.join("logs", "marl_history.jsonl"), "a", encoding="utf-8") as f_hist:
                    f_hist.write(json.dumps(metrics) + "\n")
            except Exception:
                pass

        if done_dict["__all__"]:
            r_str = " | ".join(f"{tid}:{r:6.1f}" for tid, r in episode_rewards.items())
            print(f"🔄 Episode Done @ Step {step:6d} | Rewards: {r_str}")
            episode_rewards = {tid: 0.0 for tid in TL_IDS}
            obs_dict, _ = marl_env.reset()
        else:
            obs_dict = next_obs_dict

        # 5. Periodic checkpoint save
        if step % 10000 == 0:
            for tl_id, model in models.items():
                model.save(os.path.join(model_dir, f"dqn_{tl_id}_step{step}.zip"))
            print(f"💾 Checkpoints saved at step {step}")

    # Final save upon completion
    for tl_id, model in models.items():
        model.save(os.path.join(model_dir, f"dqn_{tl_id}.zip"))
    print("\n🎉 MARL 2x2 Grid Training completed successfully! Final models saved in models/marl/")
    marl_env.close()


if __name__ == "__main__":
    train_marl()
