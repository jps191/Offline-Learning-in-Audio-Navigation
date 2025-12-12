import numpy as np
from typing import Dict, List, NamedTuple, cast, Tuple, Optional
import attr
import time

from mlagents.torch_utils import torch, nn, default_device

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.networks import ValueNetwork, SharedActorCritic
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil, AgentBufferField
from mlagents_envs.timers import timed
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.settings import TrainerSettings, OffPolicyHyperparamSettings
from contextlib import ExitStack
from mlagents.trainers.trajectory import ObsUtil

EPSILON = 1e-6  # Small value to avoid divide by zero

logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class TD3Settings(OffPolicyHyperparamSettings):
    batch_size: int = 128
    buffer_size: int = 50000
    buffer_init_steps: int = 0
    tau: float = 0.005
    steps_per_update: float = 1
    save_replay_buffer: bool = False
    policy_delay: int = 2  # TD3-specific: delay policy updates
    target_noise: float = 0.2  # TD3-specific: target policy smoothing noise
    target_noise_clip: float = 0.5  # TD3-specific: clip range for target noise
    exploration_noise: float = 0.1  # TD3-specific: Gaussian noise std for exploration during training
    reward_signal_steps_per_update: float = attr.ib()

    @reward_signal_steps_per_update.default
    def _reward_signal_steps_per_update_default(self):
        return self.steps_per_update


class TorchTD3Optimizer(TorchOptimizer):
    class PolicyValueNetwork(nn.Module):
        def __init__(
            self,
            stream_names: List[str],
            observation_specs: List[ObservationSpec],
            network_settings: NetworkSettings,
            action_spec: ActionSpec,
        ):
            super().__init__()
            num_value_outs = 1  # TD3 outputs single Q-value for continuous actions
            num_action_ins = int(action_spec.continuous_size)

            self.q1_network = ValueNetwork(
                stream_names,
                observation_specs,
                network_settings,
                num_action_ins,
                num_value_outs,
            )
            self.q2_network = ValueNetwork(
                stream_names,
                observation_specs,
                network_settings,
                num_action_ins,
                num_value_outs,
            )

        def forward(
            self,
            inputs: List[torch.Tensor],
            actions: Optional[torch.Tensor] = None,
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
            q1_grad: bool = True,
            q2_grad: bool = True,
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
            """
            Performs a forward pass on the value network, which consists of a Q1 and Q2
            network. Optionally does not evaluate gradients for either the Q1, Q2, or both.
            :param inputs: List of observation tensors.
            :param actions: For a continuous Q function (has actions), tensor of actions.
                Otherwise, None.
            :param memories: Initial memories if using memory. Otherwise, None.
            :param sequence_length: Sequence length if using memory.
            :param q1_grad: Whether or not to compute gradients for the Q1 network.
            :param q2_grad: Whether or not to compute gradients for the Q2 network.
            :return: Tuple of two dictionaries, which both map {reward_signal: Q} for Q1 and Q2,
                respectively.
            """
            # ExitStack allows us to enter the torch.no_grad() context conditionally
            with ExitStack() as stack:
                if not q1_grad:
                    stack.enter_context(torch.no_grad())
                q1_out, _ = self.q1_network(
                    inputs,
                    actions=actions,
                    memories=memories,
                    sequence_length=sequence_length,
                )
            with ExitStack() as stack:
                if not q2_grad:
                    stack.enter_context(torch.no_grad())
                q2_out, _ = self.q2_network(
                    inputs,
                    actions=actions,
                    memories=memories,
                    sequence_length=sequence_length,
                )
            return q1_out, q2_out

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]
        if isinstance(policy.actor, SharedActorCritic):
            raise UnityTrainerException("TD3 does not support SharedActorCritic")
        
        hyperparameters: TD3Settings = cast(
            TD3Settings, trainer_settings.hyperparameters
        )

        self.tau = hyperparameters.tau
        self.policy_delay = hyperparameters.policy_delay
        self.target_noise = hyperparameters.target_noise
        self.target_noise_clip = hyperparameters.target_noise_clip
        self.exploration_noise = hyperparameters.exploration_noise

        self.policy = policy
        policy_network_settings = policy.network_settings

        self.burn_in_ratio = 0.0

        self.stream_names = list(self.reward_signals.keys())
        # Use to reduce "survivor bonus" when using Curiosity or GAIL.
        self.gammas = [_val.gamma for _val in trainer_settings.reward_signals.values()]
        self.use_dones_in_backup = {
            name: int(not self.reward_signals[name].ignore_done)
            for name in self.stream_names
        }
        self._action_spec = self.policy.behavior_spec.action_spec

        # TD3 only works with continuous actions
        if self._action_spec.discrete_size > 0:
            raise UnityTrainerException(
                "TD3 only supports continuous action spaces. Use SAC or PPO for discrete actions."
            )

        self.q_network = TorchTD3Optimizer.PolicyValueNetwork(
            self.stream_names,
            self.policy.behavior_spec.observation_specs,
            policy_network_settings,
            self._action_spec,
        )

        # TD3 uses target Q-networks and target policy
        self.target_q_network = TorchTD3Optimizer.PolicyValueNetwork(
            self.stream_names,
            self.policy.behavior_spec.observation_specs,
            policy_network_settings,
            self._action_spec,
        )
        # Initialize target networks with same weights
        ModelUtils.soft_update(self.q_network.q1_network, self.target_q_network.q1_network, 1.0)
        ModelUtils.soft_update(self.q_network.q2_network, self.target_q_network.q2_network, 1.0)
        # Initialize target networks with same weights
        ModelUtils.soft_update(self.q_network.q1_network, self.target_q_network.q1_network, 1.0)
        ModelUtils.soft_update(self.q_network.q2_network, self.target_q_network.q2_network, 1.0)

        policy_params = list(self.policy.actor.parameters())
        value_params = list(self.q_network.parameters())

        logger.debug("value_vars")
        for param in value_params:
            logger.debug(param.shape)
        logger.debug("policy_vars")
        for param in policy_params:
            logger.debug(param.shape)

        self.decay_learning_rate = ModelUtils.DecayedValue(
            hyperparameters.learning_rate_schedule,
            hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.policy_optimizer = torch.optim.Adam(
            policy_params, lr=hyperparameters.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            value_params, lr=hyperparameters.learning_rate
        )
        self._move_to_device(default_device())
        
        # Track update timing
        self._last_update_time = time.time()
        self._update_start_time = None
        # TD3 update counter for delayed policy updates
        self._update_count = 0

    def get_trajectory_value_estimates(
        self,
        batch: AgentBuffer,
        next_obs: List[np.ndarray],
        done: bool,
        agent_id: str = "",
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Optional[AgentBufferField]]:
        """
        For TD3, we need to compute Q-values with actions from the buffer.
        Override to use Q1 network with actual actions instead of value network.
        """
        n_obs = len(self.policy.behavior_spec.observation_specs)

        # Convert to tensors
        current_obs = [
            ModelUtils.list_to_tensor(obs) for obs in ObsUtil.from_buffer(batch, n_obs)
        ]
        next_obs_tensors = [ModelUtils.list_to_tensor(obs) for obs in next_obs]
        next_obs_tensors = [obs.unsqueeze(0) for obs in next_obs_tensors]

        # Get actions from the buffer
        actions = AgentAction.from_buffer(batch)
        cont_actions = actions.continuous_tensor

        # Use Q1 network to get value estimates with actual actions
        with torch.no_grad():
            q1_out, _ = self.q_network.q1_network(
                current_obs,
                actions=cont_actions,
                memories=None,
                sequence_length=batch.num_experiences
            )
            
            # Get next Q-values for bootstrapping
            # Use policy to get next actions
            next_actions, _, _ = self.policy.actor.get_action_and_stats(
                next_obs_tensors,
                masks=None,
                memories=None,
                sequence_length=1
            )
            next_cont_actions = next_actions.continuous_tensor
            
            next_q1_out, _ = self.q_network.q1_network(
                next_obs_tensors,
                actions=next_cont_actions,
                memories=None,
                sequence_length=1
            )

        # Convert to numpy
        value_estimates = {}
        next_value_estimate = {}
        for name in q1_out.keys():
            value_estimates[name] = ModelUtils.to_numpy(q1_out[name])
            next_value_estimate[name] = ModelUtils.to_numpy(next_q1_out[name])

        if done:
            for k in next_value_estimate:
                if not self.reward_signals[k].ignore_done:
                    next_value_estimate[k] = 0.0

        return value_estimates, next_value_estimate, None

    def add_exploration_noise(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to actions for exploration during training.
        :param actions: The actions tensor to add noise to
        :return: Noisy actions clipped to [-1, 1]
        """
        noise = torch.randn_like(actions) * self.exploration_noise
        noisy_actions = torch.clamp(actions + noise, -1.0, 1.0)
        return noisy_actions

    @property
    def critic(self):
        return self.q_network.q1_network

    def _move_to_device(self, device: torch.device) -> None:
        self.target_q_network.to(device)
        self.q_network.to(device)

    def td3_q_loss(
        self,
        q1_out: Dict[str, torch.Tensor],
        q2_out: Dict[str, torch.Tensor],
        target_values: Dict[str, torch.Tensor],
        dones: torch.Tensor,
        rewards: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TD3 Q-loss without entropy terms"""
        q1_losses = []
        q2_losses = []
        # Multiple q losses per stream
        for i, name in enumerate(q1_out.keys()):
            q1_stream = q1_out[name].squeeze()
            q2_stream = q2_out[name].squeeze()
            with torch.no_grad():
                q_backup = rewards[name] + (
                    (1.0 - self.use_dones_in_backup[name] * dones)
                    * self.gammas[i]
                    * target_values[name]
                )
            _q1_loss = 0.5 * ModelUtils.masked_mean(
                torch.nn.functional.mse_loss(q1_stream, q_backup), loss_masks
            )
            _q2_loss = 0.5 * ModelUtils.masked_mean(
                torch.nn.functional.mse_loss(q2_stream, q_backup), loss_masks
            )

            q1_losses.append(_q1_loss)
            q2_losses.append(_q2_loss)
        q1_loss = torch.mean(torch.stack(q1_losses))
        q2_loss = torch.mean(torch.stack(q2_losses))
        return q1_loss, q2_loss

    def td3_policy_loss(
        self,
        q1_outs: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
    ) -> torch.Tensor:
        """TD3 policy loss - simple deterministic policy gradient"""
        mean_q1 = torch.mean(torch.stack(list(q1_outs.values())), axis=0)
        # TD3: maximize Q-value (minimize negative Q)
        policy_loss = ModelUtils.masked_mean(-mean_q1, loss_masks)
        return policy_loss

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Updates model using buffer with TD3 algorithm.
        :param num_sequences: Number of trajectories in batch.
        :param batch: Experience mini-batch.
        :return: Output from update process.
        """
        self._update_start_time = time.time()
        rewards = {}
        for name in self.reward_signals:
            rewards[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.rewards_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        next_obs = ObsUtil.from_buffer_next(batch, n_obs)
        # Convert to tensors
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        memories_list = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]

        if len(memories_list) > 0:
            memories = torch.stack(memories_list).unsqueeze(0)
        else:
            memories = None

        # Q network memories are 0'ed out, since we don't have them during inference.
        q_memories = (
            torch.zeros_like(memories) if memories is not None else None
        )

        # Copy normalizers from policy
        self.q_network.q1_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.q_network.q2_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.target_q_network.q1_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.target_q_network.q2_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )

        cont_actions = actions.continuous_tensor

        # Compute Q-values for current state-action pairs
        q1_out, q2_out = self.q_network(
            current_obs,
            cont_actions,
            memories=q_memories,
            sequence_length=self.policy.sequence_length,
        )

        # Compute target Q-values with target policy smoothing (TD3 trick)
        with torch.no_grad():
            # Get next actions from policy actor (deterministic)
            # For TD3, we use the current policy to generate target actions
            encoding, _ = self.policy.actor.network_body(
                next_obs, memories=memories, sequence_length=1
            )
            next_actions, _, _ = self.policy.actor.action_model.forward(encoding, act_masks)
            next_cont_actions = next_actions.continuous_tensor
            
            # Add clipped noise to target actions (target policy smoothing)
            noise = torch.randn_like(next_cont_actions) * self.target_noise
            noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
            next_cont_actions = torch.clamp(next_cont_actions + noise, -1.0, 1.0)
            
            # Compute target Q-values
            target_q1_out, target_q2_out = self.target_q_network(
                next_obs,
                next_cont_actions,
                memories=q_memories,
                sequence_length=self.policy.sequence_length,
            )
            
            # Take minimum of two target Q-values (clipped double Q-learning)
            target_values = {}
            for name in target_q1_out.keys():
                target_values[name] = torch.min(
                    target_q1_out[name], target_q2_out[name]
                ).squeeze()

        masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)
        dones = ModelUtils.list_to_tensor(batch[BufferKey.DONE])

        # Compute Q-losses
        q1_loss, q2_loss = self.td3_q_loss(
            q1_out, q2_out, target_values, dones, rewards, masks
        )
        total_value_loss = q1_loss + q2_loss

        # Update Q-networks
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        ModelUtils.update_learning_rate(self.value_optimizer, decay_lr)
        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        self.value_optimizer.step()

        # Delayed policy updates (TD3 trick)
        self._update_count += 1
        policy_loss_val = 0.0
        
        if self._update_count % self.policy_delay == 0:
            # Compute policy actions (deterministic)
            encoding, _ = self.policy.actor.network_body(
                current_obs, memories=memories, sequence_length=1
            )
            policy_actions, _, _ = self.policy.actor.action_model.forward(encoding, act_masks)
            policy_cont_actions = policy_actions.continuous_tensor
            
            # Compute Q1 values for policy actions
            q1p_out, _ = self.q_network(
                current_obs,
                policy_cont_actions,
                memories=q_memories,
                sequence_length=self.policy.sequence_length,
                q2_grad=False,  # Don't compute gradients for Q2
            )
            
            # Compute policy loss
            policy_loss = self.td3_policy_loss(q1p_out, masks)
            policy_loss_val = policy_loss.item()
            
            # Update policy
            ModelUtils.update_learning_rate(self.policy_optimizer, decay_lr)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Update target networks (TD3 delayed updates)
            ModelUtils.soft_update(self.q_network.q1_network, self.target_q_network.q1_network, self.tau)
            ModelUtils.soft_update(self.q_network.q2_network, self.target_q_network.q2_network, self.tau)

        update_stats = {
            "Losses/Policy Loss": policy_loss_val,
            "Losses/Q1 Loss": q1_loss.item(),
            "Losses/Q2 Loss": q2_loss.item(),
            "Policy/Learning Rate": decay_lr,
        }

        return update_stats

    def get_modules(self):
        modules = {
            "Optimizer:q_network": self.q_network,
            "Optimizer:target_q_network": self.target_q_network,
            "Optimizer:policy_optimizer": self.policy_optimizer,
            "Optimizer:value_optimizer": self.value_optimizer,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules
