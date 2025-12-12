# ## ML-Agent Learning (TD3)
# Contains an implementation of TD3 as described in https://arxiv.org/abs/1802.09477
# Twin Delayed Deep Deterministic Policy Gradient

from typing import cast

import numpy as np
import torch

from mlagents_envs.logging_util import get_logger
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.buffer import BufferKey
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.trainer.off_policy_trainer import OffPolicyTrainer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.policy.policy import Policy
from mlagents.trainers.td3.optimizer_torch import TorchTD3Optimizer, TD3Settings
from mlagents.trainers.trajectory import Trajectory, ObsUtil
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings

from mlagents.trainers.torch_entities.networks import SimpleActor

logger = get_logger(__name__)

BUFFER_TRUNCATE_PERCENT = 0.8

TRAINER_NAME = "td3"


class TD3Policy(TorchPolicy):
    """
    Custom policy for TD3 that adds exploration noise during training.
    """
    def __init__(
        self,
        seed: int,
        behavior_spec: BehaviorSpec,
        network_settings,
        actor_cls: type,
        actor_kwargs: dict,
        exploration_noise: float = 0.1,
    ):
        super().__init__(seed, behavior_spec, network_settings, actor_cls, actor_kwargs)
        self.exploration_noise = exploration_noise
        self._is_training = True  # Track if we're in training mode

    def set_training_mode(self, training: bool):
        """Set whether the policy is in training mode (adds noise) or evaluation mode."""
        self._is_training = training

    def evaluate(self, decision_requests, global_agent_ids):
        """
        Override evaluate to add exploration noise during training for TD3.
        """
        run_out = super().evaluate(decision_requests, global_agent_ids)
        
        # Add Gaussian noise to continuous actions during training
        if self._is_training and self.exploration_noise > 0:
            action_tuple = run_out.get("action")
            if action_tuple is not None and action_tuple.continuous is not None:
                continuous_actions = torch.as_tensor(action_tuple.continuous)
                noise = torch.randn_like(continuous_actions) * self.exploration_noise
                noisy_actions = torch.clamp(continuous_actions + noise, -1.0, 1.0)
                
                # Update the action tuple with noisy actions
                # Move to CPU if on CUDA before converting to numpy
                noisy_actions_numpy = noisy_actions.cpu().numpy() if noisy_actions.is_cuda else noisy_actions.numpy()
                noisy_action_tuple = type(action_tuple)(
                    continuous=noisy_actions_numpy,
                    discrete=action_tuple.discrete
                )
                run_out["action"] = noisy_action_tuple
        
        return run_out


class TD3Trainer(OffPolicyTrainer):
    """
    The TD3Trainer is an implementation of the TD3 algorithm.
    TD3 is designed for continuous action spaces only.
    """

    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        """
        Responsible for collecting experiences and training TD3 model.
        :param behavior_name: The name of the behavior associated with trainer config
        :param reward_buff_cap: Max reward history to track in the reward buffer
        :param trainer_settings: The parameters for the trainer.
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param artifact_path: The directory within which to store artifacts from this trainer.
        """
        super().__init__(
            behavior_name,
            reward_buff_cap,
            trainer_settings,
            training,
            load,
            seed,
            artifact_path,
        )

        self.seed = seed
        self.policy: TD3Policy = None  # type: ignore
        self.optimizer: TorchTD3Optimizer = None  # type: ignore
        self.hyperparameters: TD3Settings = cast(
            TD3Settings, trainer_settings.hyperparameters
        )
        self._step = 0

        # Don't divide by zero
        self.update_steps = 1
        self.reward_signal_update_steps = 1

        self.steps_per_update = self.hyperparameters.steps_per_update
        self.reward_signal_steps_per_update = (
            self.hyperparameters.reward_signal_steps_per_update
        )

        self.checkpoint_replay_buffer = self.hyperparameters.save_replay_buffer

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Takes a trajectory and processes it, putting it into the replay buffer.
        """
        super()._process_trajectory(trajectory)
        last_step = trajectory.steps[-1]
        agent_id = trajectory.agent_id  # All the agents should have the same ID

        agent_buffer_trajectory = trajectory.to_agentbuffer()
        # Check if we used group rewards, warn if so.
        self._warn_if_group_reward(agent_buffer_trajectory)

        # Update the normalization
        if self.is_training:
            self.policy.actor.update_normalization(agent_buffer_trajectory)
            self.optimizer.critic.update_normalization(agent_buffer_trajectory)

        # Evaluate all reward functions for reporting purposes
        self.collected_rewards["environment"][agent_id] += np.sum(
            agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS]
        )
        for name, reward_signal in self.optimizer.reward_signals.items():
            evaluate_result = (
                reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
            )

            # Report the reward signals
            self.collected_rewards[name][agent_id] += np.sum(evaluate_result)

        # Get all value estimates for reporting purposes
        (
            value_estimates,
            _,
            value_memories,
        ) = self.optimizer.get_trajectory_value_estimates(
            agent_buffer_trajectory, trajectory.next_obs, trajectory.done_reached
        )
        if value_memories is not None:
            agent_buffer_trajectory[BufferKey.CRITIC_MEMORY].set(value_memories)

        for name, v in value_estimates.items():
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Value",
                np.mean(v),
            )

        # Bootstrap using the last step rather than the bootstrap step if max step is reached.
        # Set last element to duplicate obs and remove dones.
        if last_step.interrupted:
            last_step_obs = last_step.obs
            for i, obs in enumerate(last_step_obs):
                agent_buffer_trajectory[ObsUtil.get_name_at_next(i)][-1] = obs
            agent_buffer_trajectory[BufferKey.DONE][-1] = False

        self._append_to_update_buffer(agent_buffer_trajectory)

        if trajectory.done_reached:
            self._update_end_episode_stats(agent_id, self.optimizer)

    def create_optimizer(self) -> TorchOptimizer:
        return TorchTD3Optimizer(  # type: ignore
            cast(TorchPolicy, self.policy), self.trainer_settings  # type: ignore
        )  # type: ignore

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        """
        Creates a policy with a PyTorch backend and TD3 hyperparameters
        :param parsed_behavior_id:
        :param behavior_spec: specifications for policy construction
        :return policy
        """
        actor_cls = SimpleActor
        actor_kwargs = {"conditional_sigma": False, "tanh_squash": True}

        policy = TD3Policy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            actor_kwargs,
            exploration_noise=self.hyperparameters.exploration_noise,
        )
        self.maybe_load_replay_buffer()
        return policy

    def get_policy(self, name_behavior_id: str) -> Policy:
        """
        Gets policy from trainer associated with name_behavior_id
        :param name_behavior_id: full identifier of policy
        """

        return self.policy

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME
