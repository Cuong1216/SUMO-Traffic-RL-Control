"""
Script build_net.py
Dùng để biên dịch các file XML (.nod, .edg, .con, .tll) thành mạng lưới SUMO chính thức (intersection.net.xml).
"""
import os
import subprocess
import sys

def get_netconvert_binary() -> str:
    import shutil
    binary_path = shutil.which("netconvert")
    if binary_path:
        return binary_path
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = os.path.join(sumo_home, "bin", "netconvert.exe")
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
        candidate = os.path.join(p, "netconvert.exe")
        if os.path.exists(candidate):
            return candidate
    return "netconvert"

def build_network():
    cfg_dir = os.path.join(os.path.dirname(__file__), "sumo_config")
    net_file = os.path.join(cfg_dir, "intersection.net.xml")
    
    netconvert_bin = get_netconvert_binary()
    
    # Lệnh netconvert
    cmd = [
        netconvert_bin,
        "--node-files", os.path.join(cfg_dir, "nodes.nod.xml"),
        "--edge-files", os.path.join(cfg_dir, "edges.edg.xml"),
        "--connection-files", os.path.join(cfg_dir, "connections.con.xml"),
        "--tllogic-files", os.path.join(cfg_dir, "tll.tll.xml"),
        "--output-file", net_file,
        "--no-warnings", "true"
    ]
    
    print("🚦 Đang biên dịch mạng lưới SUMO ngã tư J0...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Đã tạo thành công file: {net_file}")
    except FileNotFoundError:
        print("❌ LỖI: Không tìm thấy lệnh 'netconvert' trong PATH.")
        print("👉 Vui lòng cài đặt SUMO (ví dụ: winget install DLR.SUMO) hoặc thêm SUMO/bin vào biến môi trường PATH.")
        print("👉 Sau đó chạy lại command: python build_net.py")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi chạy netconvert:\n{e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    build_network()
