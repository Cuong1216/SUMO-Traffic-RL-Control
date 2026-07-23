import time
from env.traffic_env import TrafficEnv
from stable_baselines3 import DQN

print("🚀 Đang khởi động kịch bản test...")

# 1. Khởi tạo môi trường và ép bật GUI
env = TrafficEnv(config_path="config.yaml")
env.use_gui = True  

# 2. Load model vừa train xong
model_path = "./models/dqn_traffic_final"
print(f"📦 Đang load model từ: {model_path}")

try:
    model = DQN.load(model_path)
    print("✅ Load model thành công!")
except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    exit()

# 3. Chạy mô phỏng
obs, info = env.reset()
print("🚦 Cửa sổ SUMO GUI đã bật! Bắt đầu mô phỏng 1000 bước...")

for step in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Chỉnh tốc độ xem (0.05 giây/bước cho dễ nhìn)
    time.sleep(0.05)
    
    if terminated or truncated:
        print(f"🏁 Kết thúc mô phỏng tại step {step}")
        break

print("👀 Giữ màn hình 5 giây...")
time.sleep(5)
env.close()