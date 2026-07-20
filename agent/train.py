"""
Training script for RL Agent (DQN) using stable-baselines3.
Optimized to run on CPU and within 8GB RAM specifications.
"""
import os
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from env.traffic_env import TrafficEnv

def train_agent(config_path="config.yaml"):
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("🚀 Initializing TrafficEnv environment...")
    env = TrafficEnv(config_path=config_path)
    
    # Create directory for logs and saved models
    log_dir = cfg["agent"]["log_dir"]
    model_dir = cfg["agent"]["model_save_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Configure CheckpointCallback to save intermediate models every 20,000 timesteps
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=model_dir,
        name_prefix="dqn_traffic_v1"
    )

    print(f"🧠 Initializing {cfg['agent']['algorithm']} algorithm (MLP Policy - CPU Only)...")
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

    print(f"▶️ Starting training for {cfg['agent']['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=cfg["agent"]["total_timesteps"],
        callback=checkpoint_callback
    )

    final_model_path = os.path.join(model_dir, "dqn_traffic_final.zip")
    model.save(final_model_path)
    print(f"🎉 Training completed successfully! Final model saved at: {final_model_path}")
    
    env.close()

if __name__ == "__main__":
    train_agent()
