"""
Training script for RL Agent (DQN) using stable-baselines3.
Optimized to run on CPU with SubprocVecEnv (multi-processing) and libsumo/traci.
"""
import os
import sys
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Ensure project root is in sys.path when script is executed directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.traffic_env import TrafficEnv

def make_env(config_path="config.yaml", rank=0, in_vec_env=False):
    def _init():
        os.environ["SUMO_WORKER_RANK"] = str(rank)
        if in_vec_env:
            os.environ["IN_VEC_ENV"] = "1"
        env = TrafficEnv(config_path=config_path)
        return env
    return _init

def train_agent(config_path="config.yaml"):
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    n_envs = cfg["agent"].get("n_envs", 1)
    print(f"🚀 Initializing TrafficEnv environment (Parallel Envs: {n_envs})...")
    
    # Pre-compile SUMO network once in main process before spawning child workers
    try:
        from build_net import build_network
        build_network()
    except Exception:
        pass

    # Kill stale SUMO processes once right at startup before spawning parallel workers
    try:
        import subprocess
        for proc_name in ["sumo-gui.exe", "sumo.exe"]:
            subprocess.run(["taskkill", "/F", "/IM", proc_name, "/T"], capture_output=True, timeout=3)
    except Exception:
        pass
    
    if n_envs > 1:
        env = SubprocVecEnv([make_env(config_path, i, in_vec_env=True) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(config_path, 0, in_vec_env=False)])
    
    # Create directory for logs and saved models
    log_dir = cfg["agent"]["log_dir"]
    model_dir = cfg["agent"]["model_save_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Configure CheckpointCallback to save intermediate models every 20,000 timesteps
    # Note: In vectorized environments, save_freq counts total steps across all envs divided by n_envs
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 20000 // max(1, n_envs)),
        save_path=model_dir,
        name_prefix="dqn_traffic_v1"
    )

    final_model_path = os.path.join(model_dir, "dqn_traffic_final.zip")

    # Check if final model exists to resume training (Transfer Learning / Fine-tuning)
    if os.path.exists(final_model_path):
        print(f"📦 Found existing model at '{final_model_path}'. Loading to resume training...")
        model = DQN.load(final_model_path, env=env, device="cpu", tensorboard_log=log_dir)
        # Set exploration rate to a low value (10%) for adaptation/fine-tuning since model is already trained
        model.exploration_rate = 0.1
        reset_num_timesteps = False
    else:
        print(f"🧠 Initializing new {cfg['agent']['algorithm']} algorithm from scratch (MLP Policy - CPU Only)...")
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=cfg["agent"]["learning_rate"],
            buffer_size=cfg["agent"]["buffer_size"],    # 50,000 to conserve RAM usage
            batch_size=cfg["agent"]["batch_size"],
            gamma=cfg["agent"]["gamma"],
            verbose=1,
            tensorboard_log=log_dir,
            device="cpu"                                # Enforce CPU processing for 8GB RAM machines
        )
        reset_num_timesteps = True

    print(f"▶️ Starting training for {cfg['agent']['total_timesteps']} timesteps across {n_envs} workers...")
    model.learn(
        total_timesteps=cfg["agent"]["total_timesteps"],
        callback=checkpoint_callback,
        reset_num_timesteps=reset_num_timesteps
    )

    model.save(final_model_path)
    print(f"🎉 Training completed successfully! Final model saved at: {final_model_path}")
    
    env.close()

if __name__ == "__main__":
    train_agent()
