# 🚦 SUMO Traffic RL Control

Dự án tối ưu hóa điều khiển đèn tín hiệu giao thông tại ngã tư tự động bằng **Học tăng cường sâu (Deep Reinforcement Learning - DQN)** kết hợp với phần mềm mô phỏng giao thông chuyên nghiệp **Eclipse SUMO**.

---

## 📋 Giới thiệu & Kiến trúc Môi trường

Hệ thống mô phỏng một ngã tư phức tạp `J0` gồm 4 hướng (Bắc, Nam, Đông, Tây) với đa dạng phương tiện (`xe máy`, `ô tô`, `xe buýt`). Môi trường được xây dựng chuẩn theo giao diện **Gymnasium** (`TrafficEnv`):

* **Ma trận quan sát 2D (State - Observation Space)**:
  * Lưới quan sát bao phủ 8 làn đường đi vào ngã tư, mỗi làn được chia thành `10 ô (cells)`, mỗi ô dài `7m` (tổng tầm nhìn `70m`).
  * Mã hóa loại phương tiện: `0 = Trống`, `1 = Xe máy`, `2 = Ô tô`, `3 = Xe buýt`.
  * Vector quan sát phẳng hoá (`dim = 82`): gồm ma trận lưới `8x10` (80 giá trị) + `1` (ID pha đèn hiện tại) + `1` (Thời gian đếm ngược còn lại).

* **Không gian hành động (Action Space)**:
  * `Discrete(2)`:
    * **`Action 0` (Giữ nguyên)**: Tiếp tục duy trì pha xanh hiện tại.
    * **`Action 1` (Chuyển pha)**: Ra lệnh chuyển sang pha vàng tiếp theo (tự động kích hoạt thời gian đếm ngược `10 giây`).

* **Hàm phần thưởng (Reward Function)**:
  $$\text{Reward} = - (\alpha \times \text{Queue Length} + \beta \times \text{Total Waiting Time})$$
  * Trong đó trọng số mặc định: $\alpha = 0.4$, $\beta = 0.6$. Mục tiêu của AI là giảm thiểu tối đa hàng đợi và thời gian chờ của phương tiện.

* **Ràng buộc an toàn (Countdown Lock)**:
  * Khi đèn chuyển sang **Pha Vàng (Yellow phase - 10 giây)**, AI bị **KHÓA** quyền can thiệp để đảm bảo an toàn giao thông. Mô phỏng tiếp tục tiến lên tự động cho đến khi hết pha vàng mới cho phép AI quyết định tiếp.

---

## 📁 Cấu trúc Thư mục Dự án

```text
Traffic Simulation/
├── agent/
│   ├── __init__.py
│   └── train.py               # Script huấn luyện AI bằng thuật toán DQN (Stable-Baselines3)
├── env/
│   ├── __init__.py
│   └── traffic_env.py         # Môi trường Gymnasium (TrafficEnv) kết nối TraCI / SUMO
├── sumo_config/
│   ├── nodes.nod.xml          # Định nghĩa nút ngã tư
│   ├── edges.edg.xml          # Định nghĩa các tuyến đường
│   ├── connections.con.xml    # Định nghĩa các hướng rẽ và kết nối làn
│   ├── tll.tll.xml            # Định nghĩa chu kỳ và pha đèn giao thông
│   ├── routes.rou.xml         # Định nghĩa luồng giao thông và các loại xe
│   ├── gui-settings.xml       # Cấu hình góc nhìn, độ thu phóng, chế độ hiển thị trên SUMO GUI
│   ├── intersection.net.xml   # Bản đồ mạng lưới giao thông (được tạo tự động từ netconvert)
│   └── simulation.sumocfg     # File cấu hình tổng hợp cho SUMO
├── build_net.py               # Script tự động biên dịch mạng lưới SUMO từ XML sang .net.xml
├── test_env.py                # Script chạy kiểm thử mô phỏng và hiển thị ma trận 2D
├── config.yaml                # File cấu hình chung (thông số env, reward, DQN hyperparameters)
├── requirements.txt           # Danh sách các thư viện Python cần thiết
└── setup.bat                  # Script tự động tạo môi trường ảo và cài đặt thư viện cho Windows
```

---

## 🛠 Hướng dẫn Cài đặt & Sử dụng

### 1. Yêu cầu Tiền đề (Prerequisites)
1. **Python 3.9 - 3.11**
2. **Eclipse SUMO Simulator** (bắt buộc phải có trong biến môi trường `PATH` hoặc `SUMO_HOME`):
   * *Cách cài đặt trên Windows qua terminal:*
     ```powershell
     winget install DLR.SUMO
     ```
   * Hoặc tải bộ cài từ: [Trang tải xuống SUMO](https://sumo.dlr.de/docs/Downloads.php)

### 2. Cài đặt Môi trường Python
Chạy file `setup.bat` (hoặc chạy các lệnh thủ công dưới đây) để tạo `venv` và cài đặt thư viện (bao gồm PyTorch CPU, Gymnasium, Stable-Baselines3, TraCI,...):
```powershell
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (PowerShell)
.\venv\Scripts\Activate.ps1

# Cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Kiểm thử Môi trường & Quan sát với SUMO GUI
Để xem xe cộ di chuyển trực quan trên màn hình mô phỏng đồ họa và in ma trận xe 2D ra Terminal:
1. Mở file `config.yaml`, đảm bảo `use_gui: true`:
   ```yaml
   env:
     use_gui: true
   ```
2. Chạy script kiểm thử:
   ```powershell
   venv\Scripts\python.exe test_env.py
   ```

### 4. Huấn luyện Mô hình AI (Training)
Để quá trình huấn luyện diễn ra với tốc độ tối đa (Headless C++ simulation, không tiêu tốn RAM/GPU cho GUI):
1. Chuyển `use_gui: false` trong file `config.yaml`:
   ```yaml
   env:
     use_gui: false
   ```
2. Chạy script huấn luyện DQN:
   ```powershell
   venv\Scripts\python.exe agent/train.py
   ```
   * Mô hình sẽ tự động lưu lại định kỳ sau mỗi 20,000 bước vào thư mục `models/` và ghi nhận log huấn luyện vào `logs/` (để xem bằng `tensorboard --logdir=logs`).

---

## 📜 Bản quyền & Giấy phép
Dự án được phát triển phục vụ mục đích nghiên cứu học tăng cường và điều khiển giao thông thông minh.
