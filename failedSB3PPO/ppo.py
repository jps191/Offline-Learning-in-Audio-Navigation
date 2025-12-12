import os
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from gymnasium.wrappers import RecordEpisodeStatistics
from gym_env_AudioRL import UnityGymWrapper

# === Load YAML config ===
yaml_path = "configuration.yaml"  # path to your YAML
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

# === Environment settings ===
env_path = config["env_settings"]["env_path"]
num_envs = config["env_settings"]["num_envs"]
base_port = config["env_settings"]["base_port"]
no_graphics = config["engine_settings"]["no_graphics"]
time_scale = config["engine_settings"].get("time_scale", 1)  # optional if your wrapper supports it

# === PPO hyperparameters ===
behavior_cfg = list(config["behaviors"].values())[0]
hp = behavior_cfg["hyperparameters"]
net_cfg = behavior_cfg["network_settings"]

# === Checkpoints ===
checkpoint_cfg = config["checkpoint_settings"]
run_id = checkpoint_cfg.get("run_id", "AudioRL_Run")
checkpoint_interval = behavior_cfg.get("checkpoint_interval", 500000)
results_dir = checkpoint_cfg.get("results_dir", "results")
os.makedirs(results_dir, exist_ok=True)

# === Reward logging callback ===
class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_path, print_freq=10000, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []
        self.print_freq = print_freq
        self.last_print = 0
        self.start_time = None

    def _on_training_start(self) -> None:
        import time
        self.start_time = time.time()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info.keys():
                self.episode_rewards.append(info["episode"]["r"])
        
        # Print progress every print_freq steps
        if self.num_timesteps - self.last_print >= self.print_freq:
            if len(self.episode_rewards) > 0:
                import time
                elapsed = time.time() - self.start_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                
                recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
                avg_reward = np.mean(recent_rewards)
                print(f"Steps: {self.num_timesteps:,} | Episodes: {len(self.episode_rewards)} | Avg Reward (last 100): {avg_reward:.2f} | Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
            self.last_print = self.num_timesteps
        
        return True

    def _on_training_end(self) -> None:
        # Save all rewards to CSV at the end of training
        df = pd.DataFrame({"episode_reward": self.episode_rewards})
        df.to_csv(self.log_path, index=False)
        print(f"Saved episode rewards to {self.log_path}")

reward_log_path = os.path.join(results_dir, f"{run_id}_episode_rewards.csv")
reward_logger = RewardLoggerCallback(reward_log_path, print_freq=10000)

# === Parallel environment factory ===
def make_env(rank):
    def _init():
        env = UnityGymWrapper(env_path, worker_id=base_port + rank, no_graphics=no_graphics)
        # Wrap with RecordEpisodeStatistics for automatic episode tracking
        env = RecordEpisodeStatistics(env)
        # Optional: set Unity time_scale if your wrapper supports it
        if hasattr(env.unwrapped.env, "set_time_scale"):
            env.unwrapped.env.set_time_scale(time_scale)
        return env
    return _init

if __name__ == "__main__":
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # === PPO model setup ===
    device = config.get("torch_settings", {}).get("device", "cpu")
    print(f"Using device: {device}")
    
    # Set up TensorBoard logging
    tensorboard_log = os.path.join(results_dir, "tensorboard_logs")

    policy_kwargs = dict(
        net_arch=[net_cfg["hidden_units"]] * net_cfg["num_layers"]
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=behavior_cfg["time_horizon"] // num_envs,
        batch_size=hp["batch_size"],
        learning_rate=hp["learning_rate"],
        gamma=behavior_cfg["reward_signals"]["extrinsic"]["gamma"],
        ent_coef=hp.get("beta", 0.001),
        clip_range=hp.get("epsilon", 0.2),
        n_epochs=hp.get("num_epoch", 3),
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=tensorboard_log
    )

    # === Checkpoint callback ===
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval // num_envs,
        save_path=results_dir,
        name_prefix=run_id
    )

    # === Train the model with both callbacks ===
    total_timesteps = behavior_cfg.get("max_steps", 10_000_000)
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, reward_logger])

    # === Save final model ===
    model.save(os.path.join(results_dir, f"{run_id}_final"))

    # === Close environments ===
    env.close()
