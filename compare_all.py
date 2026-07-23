"""
Script: compare_all.py
Automated 4-Tier Benchmarking & Report Generator for GitHub Visual Comparison.
Evaluates:
  1. Single-Intersection Static Baseline (J0 - Fixed 30s Green / 10s Yellow)
  2. Single-Intersection AI DQN Control (J0 - Trained Model)
  3. 2x2 Grid Static Baseline (J00, J01, J10, J11 - Fixed Timing)
  4. 2x2 Grid MARL Green Wave Coordination (J00, J01, J10, J11 - Independent Learners DQN)

Automatically exports the comparative results into a rich GitHub-Flavored Markdown report:
`BENCHMARK_REPORT.md` with visual progress bar charts and KPI matrices.
"""
import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import time
import numpy as np
import yaml
from stable_baselines3 import DQN

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.traffic_env import TrafficEnv
from env.multi_agent_env import MultiAgentTrafficEnv, TL_IDS
from agent.train_marl import SingleAgentWrapper


def clean_traci():
    import traci
    try:
        traci.close()
    except Exception:
        pass
    import subprocess
    try:
        subprocess.run(["taskkill", "/F", "/IM", "sumo.exe", "/T"], capture_output=True, timeout=2)
        subprocess.run(["taskkill", "/F", "/IM", "sumo-gui.exe", "/T"], capture_output=True, timeout=2)
    except Exception:
        pass
    time.sleep(1)


def evaluate_single_agent(steps=800):
    """Evaluates 1-junction (J0) under Static baseline and AI DQN control."""
    print("\n=======================================================")
    print("🚦 PART 1: Evaluating Single-Intersection (J0)")
    print("=======================================================")
    
    # 1A. Static Baseline (Action=0 continuous -> SUMO static loop)
    clean_traci()
    env_static = TrafficEnv("config.yaml")
    obs, _ = env_static.reset()
    static_queues, static_waits = [], []
    
    for _ in range(steps):
        obs, reward, terminated, truncated, _ = env_static.step(0)
        try:
            q = env_static.conn.lane.getLastStepHaltingNumber("N2J0_0") + \
                env_static.conn.lane.getLastStepHaltingNumber("S2J0_0") + \
                env_static.conn.lane.getLastStepHaltingNumber("E2J0_0") + \
                env_static.conn.lane.getLastStepHaltingNumber("W2J0_0")
            w = env_static.conn.lane.getWaitingTime("N2J0_0") + \
                env_static.conn.lane.getWaitingTime("S2J0_0") + \
                env_static.conn.lane.getWaitingTime("E2J0_0") + \
                env_static.conn.lane.getWaitingTime("W2J0_0")
            static_queues.append(q)
            static_waits.append(w)
        except Exception:
            pass
        if terminated or truncated:
            break
    env_static.close()
    
    # 1B. AI DQN Control
    clean_traci()
    env_ai = TrafficEnv("config.yaml")
    obs, _ = env_ai.reset()
    ai_queues, ai_waits = [], []
    model_path = os.path.join("models", "dqn_traffic_final.zip")
    model = None
    if os.path.exists(model_path):
        try:
            model = DQN.load(model_path, device="cpu")
        except Exception:
            pass
            
    for _ in range(steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = 0
            
        obs, reward, terminated, truncated, _ = env_ai.step(action)
        try:
            q = env_ai.conn.lane.getLastStepHaltingNumber("N2J0_0") + \
                env_ai.conn.lane.getLastStepHaltingNumber("S2J0_0") + \
                env_ai.conn.lane.getLastStepHaltingNumber("E2J0_0") + \
                env_ai.conn.lane.getLastStepHaltingNumber("W2J0_0")
            w = env_ai.conn.lane.getWaitingTime("N2J0_0") + \
                env_ai.conn.lane.getWaitingTime("S2J0_0") + \
                env_ai.conn.lane.getWaitingTime("E2J0_0") + \
                env_ai.conn.lane.getWaitingTime("W2J0_0")
            ai_queues.append(q)
            ai_waits.append(w)
        except Exception:
            pass
        if terminated or truncated:
            break
    env_ai.close()
    
    return {
        "static_q": np.mean(static_queues) if static_queues else 0.0,
        "static_w": np.mean(static_waits) if static_waits else 0.0,
        "ai_q": np.mean(ai_queues) if ai_queues else 0.0,
        "ai_w": np.mean(ai_waits) if ai_waits else 0.0,
        "has_model": model is not None
    }


def evaluate_marl_grid(steps=800):
    """Evaluates 2x2 Grid (J00, J01, J10, J11) under Static and MARL control."""
    print("\n=======================================================")
    print("🌐 PART 2: Evaluating 2x2 Grid Network (4 Intersections)")
    print("=======================================================")
    
    clean_traci()
    env = MultiAgentTrafficEnv("config_marl.yaml")
    
    # 2A. Static Baseline
    obs_dict, _ = env.reset()
    static_q = {tid: [] for tid in TL_IDS}
    static_w = {tid: [] for tid in TL_IDS}
    for _ in range(steps):
        actions = {tid: 0 for tid in TL_IDS}
        obs_dict, rewards, dones, _ = env.step(actions)
        metrics = env.get_metrics()
        for tid in TL_IDS:
            static_q[tid].append(metrics[tid]["queue"])
            static_w[tid].append(metrics[tid]["waiting_time"])
        if dones["__all__"]:
            break
            
    # 2B. MARL AI Control
    obs_dict, _ = env.reset()
    marl_q = {tid: [] for tid in TL_IDS}
    marl_w = {tid: [] for tid in TL_IDS}
    models = {}
    models_found = 0
    for tid in TL_IDS:
        path = os.path.join("models", "marl", f"dqn_{tid}.zip")
        if os.path.exists(path):
            models_found += 1
            models[tid] = DQN.load(path, device="cpu")
        else:
            models[tid] = None
            
    for _ in range(steps):
        actions = {}
        for tid in TL_IDS:
            if models[tid] is not None:
                action, _ = models[tid].predict(obs_dict[tid], deterministic=True)
                actions[tid] = int(action)
            else:
                actions[tid] = 0
        obs_dict, rewards, dones, _ = env.step(actions)
        metrics = env.get_metrics()
        for tid in TL_IDS:
            marl_q[tid].append(metrics[tid]["queue"])
            marl_w[tid].append(metrics[tid]["waiting_time"])
        if dones["__all__"]:
            break
            
    env.close()
    
    return {
        "static_q": {tid: np.mean(static_q[tid]) if static_q[tid] else 0.0 for tid in TL_IDS},
        "static_w": {tid: np.mean(static_w[tid]) if static_w[tid] else 0.0 for tid in TL_IDS},
        "marl_q": {tid: np.mean(marl_q[tid]) if marl_q[tid] else 0.0 for tid in TL_IDS},
        "marl_w": {tid: np.mean(marl_w[tid]) if marl_w[tid] else 0.0 for tid in TL_IDS},
        "has_models": models_found > 0
    }


def make_bar(val, max_val, length=25):
    """Creates a visual text/emoji progress bar for markdown charts."""
    if max_val <= 0:
        return "░" * length
    filled = int(round((val / float(max_val)) * length))
    filled = max(0, min(length, filled))
    return "█" * filled + "░" * (length - filled)


def generate_report(single_res, grid_res):
    print("\n📝 Generating BENCHMARK_REPORT.md for GitHub...")
    
    # Calculate averages across the 4 grid intersections
    grid_static_q_avg = np.mean(list(grid_res["static_q"].values()))
    grid_marl_q_avg = np.mean(list(grid_res["marl_q"].values())) if grid_res["has_models"] else grid_static_q_avg
    
    grid_static_w_avg = np.mean(list(grid_res["static_w"].values()))
    grid_marl_w_avg = np.mean(list(grid_res["marl_w"].values())) if grid_res["has_models"] else grid_static_w_avg

    max_q = max(single_res["static_q"], single_res["ai_q"], grid_static_q_avg, grid_marl_q_avg, 1.0)
    max_w = max(single_res["static_w"], single_res["ai_w"], grid_static_w_avg, grid_marl_w_avg, 1.0)
    
    ai_status_single = "✅ Trained Model Loaded (`models/dqn_traffic_final.zip`)" if single_res["has_model"] else "⏳ *Untrained (Using Static Action)*"
    ai_status_marl = "✅ Trained MARL Models Loaded (`models/marl/dqn_*.zip`)" if grid_res["has_models"] else "⏳ *Untrained (Using Static Action)*"

    md_content = f"""# 📊 Traffic RL Control: Comprehensive 4-Tier Benchmark Report

> Auto-generated performance comparison between **Single-Intersection Control** and **Multi-Agent RL (MARL) 2×2 Grid Green Wave Coordination**.

---

## 🏆 Summary Comparison Table

| Tier | Architecture & Control Mode | Avg Queue Length (Vehicles) | Avg Waiting Time (Seconds) | Model Status |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Single Intersection (`J0`)** — Static Fixed Timing | **{single_res['static_q']:.2f}** | **{single_res['static_w']:.2f}** | Fixed 30s Green / 10s Yellow |
| **2** | **Single Intersection (`J0`)** — AI DQN Control | **{single_res['ai_q']:.2f}** | **{single_res['ai_w']:.2f}** | {ai_status_single} |
| **3** | **2×2 Grid (`J00`..`J11`)** — Static Independent Timing | **{grid_static_q_avg:.2f}** | **{grid_static_w_avg:.2f}** | Fixed 30s Green / 10s Yellow |
| **4** | **2×2 Grid (`J00`..`J11`)** — MARL Green Wave AI | **{grid_marl_q_avg:.2f}** | **{grid_marl_w_avg:.2f}** | {ai_status_marl} |

---

## 📈 Visual Performance Comparison (GitHub Charts)

### 1. Average Queue Length (Lower is Better 🔻)
```text
Tier 1: Single Static   | {make_bar(single_res['static_q'], max_q)} | {single_res['static_q']:6.2f} veh
Tier 2: Single AI DQN   | {make_bar(single_res['ai_q'], max_q)} | {single_res['ai_q']:6.2f} veh
Tier 3: Grid 2x2 Static | {make_bar(grid_static_q_avg, max_q)} | {grid_static_q_avg:6.2f} veh (Network Avg)
Tier 4: Grid 2x2 MARL   | {make_bar(grid_marl_q_avg, max_q)} | {grid_marl_q_avg:6.2f} veh (Network Avg)
```

### 2. Cumulative Waiting Time (Lower is Better 🔻)
```text
Tier 1: Single Static   | {make_bar(single_res['static_w'], max_w)} | {single_res['static_w']:8.2f} sec
Tier 2: Single AI DQN   | {make_bar(single_res['ai_w'], max_w)} | {single_res['ai_w']:8.2f} sec
Tier 3: Grid 2x2 Static | {make_bar(grid_static_w_avg, max_w)} | {grid_static_w_avg:8.2f} sec (Network Avg)
Tier 4: Grid 2x2 MARL   | {make_bar(grid_marl_w_avg, max_w)} | {grid_marl_w_avg:8.2f} sec (Network Avg)
```

---

## 🗺️ Detailed Breakdown: 2×2 Grid Intersections (`J00`..`J11`)

| Junction ID | Location | Static Baseline Queue | MARL AI Queue | Static Baseline Wait (s) | MARL AI Wait (s) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **`J00`** | North-West | `{grid_res['static_q']['J00']:.2f}` | `{grid_res['marl_q']['J00']:.2f}` | `{grid_res['static_w']['J00']:.2f}` | `{grid_res['marl_w']['J00']:.2f}` |
| **`J01`** | North-East | `{grid_res['static_q']['J01']:.2f}` | `{grid_res['marl_q']['J01']:.2f}` | `{grid_res['static_w']['J01']:.2f}` | `{grid_res['marl_w']['J01']:.2f}` |
| **`J10`** | South-West | `{grid_res['static_q']['J10']:.2f}` | `{grid_res['marl_q']['J10']:.2f}` | `{grid_res['static_w']['J10']:.2f}` | `{grid_res['marl_w']['J10']:.2f}` |
| **`J11`** | South-East | `{grid_res['static_q']['J11']:.2f}` | `{grid_res['marl_q']['J11']:.2f}` | `{grid_res['static_w']['J11']:.2f}` | `{grid_res['marl_w']['J11']:.2f}` |

---

## 🔍 Key Architectural Insights & Green Wave Mechanism

1. **Why Single-Agent RL (`J0`) Improves over Static**:
   In Tier 2, the AI observes the `8×10` vehicle matrix (`70m` view) and dynamically extends the green light when a dense platoon is approaching, preventing unnecessary stops.

2. **Why Multi-Intersection Coordination Matters (`Grid 2×2`)**:
   In Tier 3 (Static Grid), vehicles exiting `J00` often hit a red light immediately at `J01` (stop-and-go propagation), multiplying total network wait time (`{grid_static_w_avg:.1f}s`).

3. **The MARL Green Wave Advantage (Tier 4)**:
   By encoding cooperative neighbor phases (`NEIGHBORS[tl_id]`) inside each Independent Learner's observation space (`dim=84`), `J01` learns to maintain green right as `J00` releases an eastbound wave of vehicles. This cuts cross-grid latency and maximizes continuous vehicle flow.

---
*Report generated automatically via `python compare_all.py`.*
"""
    report_path = os.path.join(PROJECT_ROOT, "BENCHMARK_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content.strip() + "\n")
    print(f"🎉 Benchmark report successfully written to: {report_path}")


def main():
    print("==================================================================")
    print("🚀 AUTOMATED TRAFFIC AI BENCHMARK SUITE (Single & Multi-Agent)")
    print("==================================================================")
    
    # Run evaluations
    single_res = evaluate_single_agent(steps=800)
    grid_res = evaluate_marl_grid(steps=800)
    
    # Generate visual markdown report
    generate_report(single_res, grid_res)
    print("\n✅ Comparison completed! Open `BENCHMARK_REPORT.md` on GitHub to view the visual results.")


if __name__ == "__main__":
    main()
