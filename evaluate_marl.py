"""
Script: evaluate_marl.py
Evaluates and compares the 2x2 Grid Traffic Network under:
1. Static Baseline Control (Fixed 30s Green / 10s Yellow cycle across J00, J01, J10, J11)
2. MARL Independent Learners Control (Trained DQN models evaluating local + neighbor cooperative states)
Outputs comparative KPIs: Queue Length, Cumulative Waiting Time, and Throughput.
"""
import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import yaml
from stable_baselines3 import DQN

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.multi_agent_env import MultiAgentTrafficEnv, TL_IDS
from agent.train_marl import SingleAgentWrapper


def evaluate_baseline(env: MultiAgentTrafficEnv, steps=1000):
    print("\n-------------------------------------------------------------")
    print("🚦 [Mode 1] Evaluating Static Baseline Control (Fixed Timing)")
    print("-------------------------------------------------------------")
    obs, _ = env.reset()
    
    total_queue = {tid: [] for tid in TL_IDS}
    total_wait = {tid: [] for tid in TL_IDS}
    total_rewards = {tid: 0.0 for tid in TL_IDS}
    
    # In static control, we send action=0 (keep phase) constantly, allowing SUMO static program
    # (30s green -> 10s yellow) to naturally loop without AI interventions.
    for step in range(1, steps + 1):
        actions = {tid: 0 for tid in TL_IDS}
        obs, rewards, dones, info = env.step(actions)
        
        for tid in TL_IDS:
            total_rewards[tid] += rewards[tid]
            
        if step % 10 == 0:
            metrics = env.get_metrics()
            for tid in TL_IDS:
                total_queue[tid].append(metrics[tid]["queue"])
                total_wait[tid].append(metrics[tid]["waiting_time"])
                
        if dones["__all__"]:
            break
            
    avg_q = {tid: np.mean(total_queue[tid]) if total_queue[tid] else 0.0 for tid in TL_IDS}
    avg_w = {tid: np.mean(total_wait[tid]) if total_wait[tid] else 0.0 for tid in TL_IDS}
    return avg_q, avg_w, total_rewards


def evaluate_marl(env: MultiAgentTrafficEnv, models: dict, steps=1000):
    print("\n-------------------------------------------------------------")
    print("🧠 [Mode 2] Evaluating MARL Independent Learners (AI Control)")
    print("-------------------------------------------------------------")
    obs_dict, _ = env.reset()
    
    total_queue = {tid: [] for tid in TL_IDS}
    total_wait = {tid: [] for tid in TL_IDS}
    total_rewards = {tid: 0.0 for tid in TL_IDS}
    
    for step in range(1, steps + 1):
        actions = {}
        for tid in TL_IDS:
            if tid in models and models[tid] is not None:
                action, _ = models[tid].predict(obs_dict[tid], deterministic=True)
                actions[tid] = int(action)
            else:
                actions[tid] = 0
                
        obs_dict, rewards, dones, info = env.step(actions)
        
        for tid in TL_IDS:
            total_rewards[tid] += rewards[tid]
            
        if step % 10 == 0:
            metrics = env.get_metrics()
            for tid in TL_IDS:
                total_queue[tid].append(metrics[tid]["queue"])
                total_wait[tid].append(metrics[tid]["waiting_time"])
                
        if dones["__all__"]:
            break
            
    avg_q = {tid: np.mean(total_queue[tid]) if total_queue[tid] else 0.0 for tid in TL_IDS}
    avg_w = {tid: np.mean(total_wait[tid]) if total_wait[tid] else 0.0 for tid in TL_IDS}
    return avg_q, avg_w, total_rewards


def main():
    config_path = "config_marl.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_dir = cfg["agent"]["model_save_dir"]
    
    # Check if models exist
    models = {}
    models_found = 0
    for tid in TL_IDS:
        path = os.path.join(model_dir, f"dqn_{tid}.zip")
        if os.path.exists(path):
            models_found += 1
            
    if models_found == 0:
        print("⚠️ No trained MARL models found in models/marl/. Running baseline evaluation only.")
        print("👉 Run `python agent/train_marl.py` to train the models first!")
        
    env = MultiAgentTrafficEnv(config_path)
    
    # 1. Evaluate Static Baseline
    base_q, base_w, base_r = evaluate_baseline(env, steps=1000)
    
    # 2. Evaluate MARL if available
    marl_q, marl_w, marl_r = {}, {}, {}
    if models_found > 0:
        for tid in TL_IDS:
            path = os.path.join(model_dir, f"dqn_{tid}.zip")
            if os.path.exists(path):
                wrapper = SingleAgentWrapper(env, tid)
                models[tid] = DQN.load(path, device="cpu")
            else:
                models[tid] = None
        marl_q, marl_w, marl_r = evaluate_marl(env, models, steps=1000)
        
    env.close()

    # Summary Report Table
    print("\n=========================================================================================")
    print("📊 MARL vs STATIC BASELINE EVALUATION REPORT (1000 Simulation Steps)")
    print("=========================================================================================")
    print(f"{'Intersection':<14} | {'Base Avg Queue':<16} | {'MARL Avg Queue':<16} | {'Base Avg Wait':<16} | {'MARL Avg Wait':<16}")
    print("-" * 89)
    for tid in TL_IDS:
        bq = base_q.get(tid, 0.0)
        mq = marl_q.get(tid, 0.0) if models_found > 0 else 0.0
        bw = base_w.get(tid, 0.0)
        mw = marl_w.get(tid, 0.0) if models_found > 0 else 0.0
        print(f"{tid:<14} | {bq:16.2f} | {mq:16.2f} | {bw:16.2f} | {mw:16.2f}")
    print("=========================================================================================\n")


if __name__ == "__main__":
    main()
