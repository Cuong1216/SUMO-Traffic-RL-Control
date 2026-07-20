"""
Script huấn luyện RL Agent (DQN) với stable-baselines3.
Tối ưu hóa chạy trên CPU và RAM 8GB.
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

    print("🚀 Khởi tạo môi trường TrafficEnv...")
    env = TrafficEnv(config_path=config_path)
    
    # Tạo thư mục log và model
    log_dir = cfg["agent"]["log_dir"]
    model_dir = cfg["agent"]["model_save_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Cấu hình Checkpoint lưu model tự động mỗi 20,000 timesteps
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=model_dir,
        name_prefix="dqn_traffic_v1"
    )

    print(f"🧠 Khởi tạo thuật toán {cfg['agent']['algorithm']} (MLP Policy - CPU Only)...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg["agent"]["learning_rate"],
        buffer_size=cfg["agent"]["buffer_size"],    # 50,000 để tiết kiệm RAM
        batch_size=cfg["agent"]["batch_size"],
        gamma=cfg["agent"]["gamma"],
        verbose=1,
        tensorboard_log=log_dir,
        device="cpu"                                # Ràng buộc chạy CPU cho máy RAM 8GB
    )

    print(f"▶️ Bắt đầu huấn luyện {cfg['agent']['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=cfg["agent"]["total_timesteps"],
        callback=checkpoint_callback
    )

    final_model_path = os.path.join(model_dir, "dqn_traffic_final.zip")
    model.save(final_model_path)
    print(f"🎉 Huấn luyện hoàn tất! Model đã được lưu tại: {final_model_path}")
    
    env.close()

if __name__ == "__main__":
    train_agent()
