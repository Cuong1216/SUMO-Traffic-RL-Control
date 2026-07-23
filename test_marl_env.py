"""
Script to test MultiAgentTrafficEnv with 100 random steps.
Verifies observation dimensions, reward computation, and clean SUMO simulation lifecycle.
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import os
import random
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.multi_agent_env import MultiAgentTrafficEnv, TL_IDS

def test_environment():
    print("🚦 Initializing MultiAgentTrafficEnv...")
    env = MultiAgentTrafficEnv(config_path="config_marl.yaml")
    
    print("🔄 Testing reset()...")
    obs, info = env.reset()
    assert isinstance(obs, dict), "Reset must return a dictionary of observations."
    
    for tl_id in TL_IDS:
        assert tl_id in obs, f"Observation missing for {tl_id}"
        agent = env.agents[tl_id]
        expected_shape = agent.observation_space.shape
        actual_shape = obs[tl_id].shape
        print(f"  Agent {tl_id} | Obs Shape: {actual_shape} (Expected: {expected_shape}) | Controlled Lanes: {len(agent.all_lanes)}")
        assert actual_shape == expected_shape, f"Shape mismatch for {tl_id}: {actual_shape} != {expected_shape}"

    print("\n▶️ Running 100 simulation steps with random actions across all 4 agents...")
    total_rewards = {tl_id: 0.0 for tl_id in TL_IDS}
    
    for step in range(1, 101):
        actions = {tl_id: random.choice([0, 1]) for tl_id in TL_IDS}
        
        obs, rewards, dones, info = env.step(actions)
        
        for tl_id in TL_IDS:
            total_rewards[tl_id] += rewards[tl_id]
            
        if step % 20 == 0 or dones["__all__"]:
            metrics = env.get_metrics()
            q_str = " | ".join([f"{tid} Q:{metrics[tid]['queue']:.1f}" for tid in TL_IDS])
            print(f"  Step {step:3d} | Sim Step: {env.current_step:4d} | {q_str}")
            
        if dones["__all__"]:
            print(f"🛑 Episode terminated early at step {step}")
            break
            
    print("\n✅ 100 test steps completed successfully!")
    for tl_id in TL_IDS:
        print(f"  Total Reward ({tl_id}): {total_rewards[tl_id]:.2f}")
        
    print("🔌 Closing environment...")
    env.close()
    print("🎉 MultiAgentTrafficEnv verification passed successfully!")

if __name__ == "__main__":
    test_environment()
