"""
Script kiểm thử và xác minh hoạt động của môi trường TrafficEnv (Milestone 1 Verification).
- Chạy môi trường trong 100 bước mô phỏng.
- In ra Ma trận lưới 2D mã hóa thực tế để người dùng trực quan thấy dữ liệu xe.
- Kiểm tra tính chất Countdown Lock (khóa 10s trong pha vàng).
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import time
import numpy as np
from env.traffic_env import TrafficEnv

def run_test():
    print("='*' * 60")
    print("🚦 KIỂM THỬ MÔI TRƯỜNG 1 NGÃ TƯ (MILESTONE 1 VERIFICATION)")
    print("='*' * 60")

    env = TrafficEnv(config_path="config.yaml")
    obs, info = env.reset()
    
    print(f"✅ Reset thành công! Kích thước vector quan sát (Observation dim): {obs.shape}")
    print(f"✅ Kích thước Action Space: {env.action_space}")
    print(f"✅ Danh sách 8 làn đường theo dõi: {env.all_lanes}")
    print("\n▶️ Bắt đầu mô phỏng 50 bước với các quyết định ngẫu nhiên và kiểm chứng Ma trận 2D:\n")

    for step in range(1, 51):
        # Chọn action: xen kẽ giữ pha và chuyển pha để test cả pha xanh lẫn pha vàng (countdown)
        if step % 15 == 0:
            action = 1  # Ra lệnh chuyển pha sang pha tiếp theo
            action_desc = "CHUYỂN PHA (Action=1)"
        else:
            action = 0  # Giữ nguyên pha hiện tại
            action_desc = "GIỮ NGUYÊN (Action=0)"

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Nếu đang bật chế độ GUI, tạo độ trễ nhỏ (0.15s) mỗi bước để người dùng quan sát rõ ràng chuyển động xe
        if env.use_gui:
            time.sleep(0.15)
        
        # Giải mã Observation lại thành grid (8x10), phase và countdown để in ra
        grid_flat = obs[:-2]
        phase, countdown = obs[-2], obs[-1]
        grid = grid_flat.reshape((8, 10)).astype(int)

        # Chỉ in chi tiết tại một số bước có xe hoặc khi chuyển pha
        total_vehicles_in_grid = np.count_nonzero(grid)
        if step % 10 == 0 or action == 1 or phase in [1, 3]:
            phase_name = {
                0: "Xanh Bắc-Nam",
                1: f"Vàng Bắc-Nam (Đếm ngược {countdown:.0f}s - AI KHÓA)",
                2: "Xanh Đông-Tây",
                3: f"Vàng Đông-Tây (Đếm ngược {countdown:.0f}s - AI KHÓA)"
            }.get(int(phase), f"Phase {phase}")

            print(f"--- [Bước {step:02d}] Action: {action_desc} | Pha đèn: {phase_name} | Reward: {reward:.2f} ---")
            print(f"   Tổng số xe trong lưới quan sát 70m: {total_vehicles_in_grid}")
            
            # In Ma trận lưới 2D đẹp mắt cho 4 hướng chính
            # Làn 0-1: Bắc vào (N2J0), Làn 2-3: Nam vào (S2J0), Làn 4-5: Đông vào (E2J0), Làn 6-7: Tây vào (W2J0)
            print("   [Ma trận xe 2D (0=Trống, 1=Xe máy, 2=Ô tô, 3=Buýt) - từ xa đến gần ngã tư]:")
            for idx, lane_id in enumerate(env.all_lanes):
                lane_cells = list(grid[idx])
                print(f"     Làn {lane_id:8s}: {lane_cells}")
            print()

        if terminated or truncated:
            print(f"🏁 Mô phỏng kết thúc tại bước {step}")
            break

    if env.use_gui:
        print("\n👀 Đang dừng 3 giây để bạn giữ màn hình quan sát ngã tư trước khi đóng SUMO GUI...")
        time.sleep(3)
    env.close()
    print("🎉 HOÀN TẤT KIỂM THỬ MILESTONE 1! Môi trường sẵn sàng cho huấn luyện RL.")

if __name__ == "__main__":
    run_test()
