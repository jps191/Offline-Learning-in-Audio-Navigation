# testSB3_AudioRL.py
import time
import numpy as np
from gym_env_AudioRL import UnityGymWrapper

def main():
    # Initialize the environment
    env = UnityGymWrapper(
        "AudioRL.exe",
    )

    print("Environment initialized.")
    print("\n=== OBSERVATION SPACE INFO ===")
    print(f"Observation shape: {env.observation_space.shape}")
    print(f"Observation dtype: {env.observation_space.dtype}")
    print(f"Total observation size: {env.observation_space.shape[0]}")
    
    print("\n=== ACTION SPACE INFO ===")
    print(f"Action shape: {env.action_space.shape}")
    print(f"Action range: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    
    obs, _ = env.reset()
    print("\n=== INITIAL OBSERVATION ===")
    print("Observation shape:", obs.shape)
    print("Observation values:", obs)
    print(f"Min value: {obs.min()}, Max value: {obs.max()}")

    total_steps = 50  # total steps to test
    step_count = 0
    episode_count = 1

    try:
        while step_count < total_steps:
            action = env.action_space.sample()  # sample random action
            obs, reward, done, trunc, info = env.step(action)
            
            # Show first few observations in detail
            if step_count < 3:
                print(f"\n[Step {step_count+1}]")
                print(f"  Action: {action}")
                print(f"  Observation: {obs}")
                print(f"  Reward: {reward}")
            else:
                print(f"[Episode {episode_count} | Step {step_count+1}] Reward: {reward}")

            step_count += 1

            # Reset environment if episode ends
            if done:
                obs, _ = env.reset()
                episode_count += 1
                print(f"\nEpisode {episode_count} started. Observation: {obs}")

            # Optional: slow down loop for readability
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Test interrupted by user.")

    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
