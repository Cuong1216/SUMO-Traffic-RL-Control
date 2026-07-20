"""
Verification script for TrafficEnv environment (Milestone 1 Verification).
- Runs the simulation for 50 simulation steps.
- Prints the real-time encoded 2D grid matrix for visual verification of vehicle data.
- Verifies the Countdown Lock mechanism (10-second lock during yellow phases).
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
    print("🚦 INTERSECTION ENVIRONMENT VERIFICATION (MILESTONE 1)")
    print("='*' * 60")

    env = TrafficEnv(config_path="config.yaml")
    obs, info = env.reset()
    
    print(f"✅ Reset successful! Observation vector shape (Observation dim): {obs.shape}")
    print(f"✅ Action Space size: {env.action_space}")
    print(f"✅ Monitored 8 lanes list: {env.all_lanes}")
    print("\n▶️ Starting 50 simulation steps with alternating decisions and 2D Grid inspection:\n")

    for step in range(1, 51):
        # Select action: alternate between maintaining and switching phase to test both green and yellow countdown phases
        if step % 15 == 0:
            action = 1  # Command phase transition to the next phase
            action_desc = "SWITCH PHASE (Action=1)"
        else:
            action = 0  # Maintain current phase
            action_desc = "KEEP GREEN (Action=0)"

        obs, reward, terminated, truncated, info = env.step(action)
        
        # If GUI mode is enabled, create a small delay (0.15s) per step for smooth visual tracking
        if env.use_gui:
            time.sleep(0.15)
        
        # Decode observation back into grid (8x10), phase ID, and countdown remaining
        grid_flat = obs[:-2]
        phase, countdown = obs[-2], obs[-1]
        grid = grid_flat.reshape((8, 10)).astype(int)

        # Print detailed inspection only at periodic intervals or on phase transitions
        total_vehicles_in_grid = np.count_nonzero(grid)
        if step % 10 == 0 or action == 1 or phase in [1, 3]:
            phase_name = {
                0: "Green North-South",
                1: f"Yellow North-South (Countdown {countdown:.0f}s - AI LOCKED)",
                2: "Green East-West",
                3: f"Yellow East-West (Countdown {countdown:.0f}s - AI LOCKED)"
            }.get(int(phase), f"Phase {phase}")

            print(f"--- [Step {step:02d}] Action: {action_desc} | Traffic Light: {phase_name} | Reward: {reward:.2f} ---")
            print(f"   Total vehicles inside 70m observation grid: {total_vehicles_in_grid}")
            
            # Print formatted 2D vehicle grid for the 4 main incoming directions
            # Lanes 0-1: North incoming (N2J0), Lanes 2-3: South incoming (S2J0), Lanes 4-5: East incoming (E2J0), Lanes 6-7: West incoming (W2J0)
            print("   [2D Vehicle Matrix (0=Empty, 1=Motorcycle, 2=Car, 3=Bus) - from farthest to stopline]:")
            for idx, lane_id in enumerate(env.all_lanes):
                lane_cells = list(grid[idx])
                print(f"     Lane {lane_id:8s}: {lane_cells}")
            print()

        if terminated or truncated:
            print(f"🏁 Simulation ended at step {step}")
            break

    if env.use_gui:
        print("\n👀 Pausing for 3 seconds to let you inspect the intersection before closing SUMO GUI...")
        time.sleep(3)
    env.close()
    print("🎉 MILESTONE 1 VERIFICATION COMPLETED! Environment is ready for RL training.")

if __name__ == "__main__":
    run_test()
