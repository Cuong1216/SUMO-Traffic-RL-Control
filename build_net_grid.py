"""
Script: build_net_grid.py
Compiles XML configuration files (.nod, .edg, .con, .tll) inside sumo_config/grid_2x2/ into grid_2x2.net.xml.
"""
import os
import subprocess
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

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

def build_grid_network():
    cfg_dir = os.path.join(os.path.dirname(__file__), "sumo_config", "grid_2x2")
    net_file = os.path.join(cfg_dir, "grid_2x2.net.xml")
    
    netconvert_bin = get_netconvert_binary()
    
    cmd = [
        netconvert_bin,
        "--node-files", os.path.join(cfg_dir, "nodes.nod.xml"),
        "--edge-files", os.path.join(cfg_dir, "edges.edg.xml"),
        "--connection-files", os.path.join(cfg_dir, "connections.con.xml"),
        "--tllogic-files", os.path.join(cfg_dir, "tll.tll.xml"),
        "--output-file", net_file,
        "--no-warnings", "true"
    ]
    
    print("🚦 Compiling SUMO 2x2 Grid network (J00, J01, J10, J11)...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Grid network file successfully generated: {net_file}")
    except FileNotFoundError:
        print("❌ ERROR: 'netconvert' executable not found in PATH.")
        print("👉 Please install SUMO (e.g. winget install DLR.SUMO) or add SUMO/bin to your PATH environment variable.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during netconvert execution:\n{e.stderr}\n{e.stdout}")
        sys.exit(1)

if __name__ == "__main__":
    build_grid_network()
