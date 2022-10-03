import numpy as np
import torch
import torch.nn.functional as F
import copy
import utils.model_rollout_functions as mrf
from utils.logger import create_stats_ordered_dict, logger
from utils.utils import print_banner


class WeightedGanACTrainer(object):

    def __init__(
            self,
            device,
            discount,                       # discount factor
            beta,                           # target network update rate
            actor_lr,                       # actor learning rate
            critic_lr,                      # critic learning rate
            dis_lr,                         # discriminator learning rate
            lmbda,                          # weight of the minimum in Q-update
            log_lagrange,                   # value of log lagrange multiplier
            policy_freq,                    # update frequency of the actor
            state_noise_std,
            num_action_bellman,
            actor,                          # Actor object
            critic,                         # Critic object
            discriminator,                  # Discriminator object
            dynamics_model,                 # Model object, Note that WeightedGanACTrainer is not responsible for training this
            replay_buffer,                  # The true replay buffer
            generated_data_buffer,          # Replay buffer solely consisting of synthetic transitions
            rollout_len_func,               # Rollout length as a function of number of train calls
            rollout_len_fix=1,              # fixed constant rollout length
            num_model_rollouts=512,         # Number of *transitions* to generate per training timestep
            rollout_generation_freq=1,      # Can save time by only generating data when model is updated
            rollout_batch_size=int(1024),   # Maximum batch size for generating rollouts (i.e. GPU memory limit)
            real_data_pct=0.05,             # Percentage of real data used for actor-critic training
            batch_size=256,
            warm_start_epochs=40,
            use_kl_dual=False,
            use_weight_wpr=True
    ):
        super().__init__()

        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=(0.4, 0.999))

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.discriminator = discriminator
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=dis_lr, betas=(0.4, 0.999))
        self.adversarial_loss = torch.nn.BCELoss(reduction='none')

        self.log_lagrange = torch.tensor(log_lagrange, device=device)

        self.dynamics_model = dynamics_model
        self.replay_buffer = replay_buffer
        self.generated_data_buffer = generated_data_buffer
        self.rollout_len_func = rollout_len_func
        self.rollout_len_fix = rollout_len_fix

        self.num_model_rollouts = num_model_rollouts
        self.rollout_generation_freq = rollout_generation_freq
        self.rollout_batch_size = rollout_batch_size
        self.real_data_pct = real_data_pct

        self.device = device
        self.discount = discount
        self.beta = beta
        self.lmbda = lmbda
        self.policy_freq = int(policy_freq)
        self.state_noise_std = state_noise_std
        self.num_action_bellman = num_action_bellman
        self.batch_size = batch_size
        self.warm_start_epochs = warm_start_epochs

        self._n_train_steps_total = 0
        self._n_epochs = 0

        self.Q_average = None
        self.epoch_critic_loss_thres = 1000.

        reward_stat = self.replay_buffer.reward_stat
        self.reward_max = reward_stat['max'] + 10. * reward_stat['std']
        self.reward_min = reward_stat['min'] - 10. * reward_stat['std']

        self.dynamics_model.set_reward_range(np.max([np.abs(self.reward_min), np.abs(self.reward_max)]))

        self.use_kl_dual = use_kl_dual
        self.use_weight_wpr = use_weight_wpr

        print_banner(f"reward_max: {self.reward_max:.3f}, reward_min: {self.reward_min:.3f}")
        print_banner(f"Initialized Model-based GAN Weighted Actor-Critic Trainer, use_kl_dual={self.use_kl_dual}, use_weight_wpr={self.use_weight_wpr} !")

    def model_based_rollout(self):
        rollout_len = self.rollout_len_func(train_steps=self._n_train_steps_total, n=self.rollout_len_fix)
        total_samples = self.rollout_generation_freq * self.num_model_rollouts

        num_samples, generated_rewards, terminated = 0, np.array([]), []
        while num_samples < total_samples:
            batch_samples = min(self.rollout_batch_size, total_samples - num_samples)
            start_states = self.replay_buffer.random_batch(batch_samples, device=self.device)['observations']       # (batch_samples, state_dim), torch.tensor

            with torch.no_grad():
                paths = mrf.policy(
                    dynamics_model=self.dynamics_model,
                    policy=self.actor,
                    start_states=start_states,
                    device=self.device,
                    max_path_length=rollout_len,
                    replay_buffer_device=self.generated_data_buffer.device
                )

            for path in paths:
                self.generated_data_buffer.add_path(path)
                num_samples += len(path['observations'])
                generated_rewards = np.concatenate([generated_rewards, path['rewards'][:, 0]]) if self.generated_data_buffer.device == "numpy" else np.concatenate([generated_rewards, path['rewards'].cpu().numpy()[:, 0]])
                terminated.append(path['terminals'][-1, 0] if self.generated_data_buffer.device == "numpy" else path['terminals'].cpu().numpy()[-1, 0])

            if num_samples >= total_samples:
                break

        return generated_rewards, terminated, rollout_len

    def sample_batch(self, n_real_data, n_generated_data):
        batch = self.replay_buffer.random_batch(n_real_data, device=self.device)
        generated_batch = self.generated_data_buffer.random_batch(n_generated_data, device=self.device)

        for k in ('rewards', 'terminals', 'observations', 'actions', 'next_observations'):
            batch[k] = torch.cat((batch[k], generated_batch[k]), dim=0)

        return batch

    def prepare_epoch_update(self):
        self._n_epochs += 1
        real_data_pct_curr = min(max(1. - (self._n_epochs - (self.warm_start_epochs + 1)) // 5 * 0.1, self.real_data_pct), 1.)
        n_real_data = int(real_data_pct_curr * self.batch_size)
        n_generated_data = self.batch_size - n_real_data
        warm_start = self._n_epochs <= self.warm_start_epochs
        return n_real_data, n_generated_data, warm_start

    def optimize_critic(self, batch, warm_start):
        with torch.no_grad():
            next_state_repeat, next_actions = self.actor_target.sample_multiple_actions(batch['next_observations'], num_action=self.num_action_bellman, std=self.state_noise_std)
            target_Q = self.critic_target.weighted_min(next_state_repeat, next_actions, lmbda=self.lmbda)
            target_Q = target_Q.view(self.batch_size, -1).mean(1).view(-1, 1)  # (batch_size, 1)
            target_Q = batch['rewards'] + (1. - batch['terminals']) * self.discount * target_Q * (target_Q.abs() < 2000.)

        current_Q1, current_Q2 = self.critic(batch['observations'], batch['actions'])
        if current_Q1.size() != target_Q.size():  # check dimensions here
            raise ValueError(f"Shape of current_Q1={current_Q1.size()}, shape of target_Q={target_Q.size()}.")

        critic_loss = F.huber_loss(current_Q1, target_Q, delta=500.) + F.huber_loss(current_Q2, target_Q, delta=500.)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.1, error_if_nonfinite=True)
        self.critic_optimizer.step()

        return target_Q, current_Q1, current_Q2, critic_loss

    def get_true_fake_samples(self):
        # (s,a) ~ d_env, true_s' ~ d_env, true_a' ~ d_env, fake_s' ~ hat{P}, fake_a' ~ pi

        samples = self.sample_batch(n_real_data=self.batch_size, n_generated_data=0)
        # By construction, observations in the generated buffer cannot be terminal state

        with torch.no_grad():
            fake_transitions = self.dynamics_model.sample(torch.cat([samples["observations"], samples["actions"]], dim=-1))
        if (fake_transitions != fake_transitions).any():
            fake_transitions[fake_transitions != fake_transitions] = 0
        fake_rewards = fake_transitions[:, :1]
        fake_delta_obs = fake_transitions[:, 1:]
        fake_next_state = samples["observations"] + fake_delta_obs
        fake_not_dones = (self.dynamics_model.termination(samples["observations"], samples["actions"], fake_next_state, fake_rewards).squeeze() < 0.5)  # check whether fake_next_state is terminal state
        fake_next_state = fake_next_state[fake_not_dones]  # only non-terminal fake_next_state can have action choice
        _, fake_next_raw_action = self.actor(fake_next_state, return_raw_action=True)       # fake_a' ~ pi

        fake_samples = torch.cat([fake_next_state, fake_next_raw_action], dim=1)  # (s_{t+1}, a_{t+1})

        true_samples = torch.cat(
            [
                samples['next_observations'][fake_not_dones], self.actor.pre_scaling_action(samples['next_actions'][fake_not_dones])
            ], dim=1
        )

        return fake_samples, true_samples, samples['weights'][fake_not_dones] * (1. - samples['terminals'][fake_not_dones])   # w: (batch_size, 1)

    def get_generator_and_discriminator_loss(self, fake_samples, true_samples, weights):
        if self.use_kl_dual:
            return self._get_generator_and_discriminator_loss_kl_dual(fake_samples, true_samples, weights, self.use_weight_wpr)
        else:
            return self._get_generator_and_discriminator_loss(fake_samples, true_samples, weights)

    def _get_generator_and_discriminator_loss_kl_dual(self, fake_samples, true_samples, weights, use_weights):
        # weights: (batch_size, 1), each (s,a) has only one (s',a')

        generator_loss = (-1.) * self.discriminator(fake_samples)

        real_loss = self.discriminator(true_samples)
        fake_loss = (-1.) * self.discriminator(fake_samples.detach())

        if use_weights:
            generator_loss = generator_loss * weights
            real_loss = real_loss * weights
            fake_loss = fake_loss * weights

        generator_loss = generator_loss.mean()
        real_loss = real_loss.mean()
        fake_loss = fake_loss.mean()
        discriminator_loss = (-1.) * (real_loss + fake_loss)

        return generator_loss, discriminator_loss, real_loss, fake_loss, {}

    def _get_generator_and_discriminator_loss(self, fake_samples, true_samples, weights):
        # weights: (batch_size, 1)

        generator_loss = self.adversarial_loss(self.discriminator(fake_samples), torch.ones(fake_samples.size(0), 1, device=self.device))
        generator_loss = (generator_loss * weights).mean()

        # Measure discriminator's ability to classify real from generated samples
        # label smoothing via soft and noisy labels
        # Uniformly random label in [lower, upper): torch.rand(size=shape, device=self.device) * (upper - lower) + lower
        fake_labels = torch.zeros(fake_samples.size(0), 1, device=self.device)
        true_labels = torch.rand(size=(true_samples.size(0), 1), device=self.device) * (1.0 - 0.80) + 0.80  # [0.80, 1.0)

        real_loss = self.adversarial_loss(self.discriminator(true_samples), true_labels)
        fake_loss = self.adversarial_loss(self.discriminator(fake_samples.detach()), fake_labels)
        real_loss = (real_loss * weights).mean()
        fake_loss = (fake_loss * weights).mean()
        discriminator_loss = (real_loss + fake_loss) / 2.

        return generator_loss, discriminator_loss, real_loss, fake_loss, {}

    def optimize_actor(self, batch, warm_start, generator_loss):
        ### Compute optimization objective for policy update ###
        Q_values = self.critic.q_min(batch['observations'], self.actor(batch['observations']))
        if self.Q_average is None:
            self.Q_average = Q_values.abs().mean().detach()

        if warm_start:
            policy_loss = generator_loss
        else:
            lagrange = self.log_lagrange / self.Q_average
            policy_loss = -lagrange * Q_values.mean() + generator_loss

        # Update policy (actor): minimize policy loss
        if self._n_train_steps_total % self.policy_freq == 0:
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

        return Q_values, policy_loss

    def optimize_discriminator(self, discriminator_loss):
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        return discriminator_loss

    def moving_average_updates(self, Q_values):
        # Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.beta * param.data + (1. - self.beta) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.beta * param.data + (1. - self.beta) * target_param.data)

        self.Q_average = self.beta * Q_values.abs().mean().detach() + (1. - self.beta) * self.Q_average

    def update_epoch_critic_loss_thres(self, epoch_critic_loss, iterations):
        assert len(epoch_critic_loss) == iterations, f"len(epoch_critic_loss)={len(epoch_critic_loss)}, should be {iterations}"
        self.epoch_critic_loss_thres = 0.95 * self.epoch_critic_loss_thres + 0.05 * (np.mean(epoch_critic_loss) + 3. * np.std(epoch_critic_loss))

    def logging_common(self, warm_start, n_real_data, n_generated_data, true_samples, target_Q, critic_loss, epoch_critic_loss, policy_loss, generator_loss, Q_values, real_loss, fake_loss, discriminator_loss, current_Q1, rollout_len, generated_rewards, terminated, weights):
        print_banner(f"Training epoch: {str(self._n_epochs)}, perform warm_start training: {warm_start}", separator="*", num_star=90)
        # Logging
        logger.record_tabular('Num Real Data', n_real_data)
        logger.record_tabular('Num Generated Data', n_generated_data)
        logger.record_tabular('Num True Samples', true_samples.shape[0])
        logger.record_dict(create_stats_ordered_dict('Q_target', target_Q.cpu().data.numpy()))
        logger.record_tabular('Critic Loss', critic_loss.cpu().data.numpy())
        logger.record_dict(create_stats_ordered_dict('Epoch Critic Loss', np.array(epoch_critic_loss)))
        logger.record_tabular('Epoch Critic Loss Thres', self.epoch_critic_loss_thres)
        logger.record_tabular('Actor Loss', policy_loss.cpu().data.numpy())
        logger.record_tabular('Generator Loss', generator_loss.cpu().data.numpy())
        logger.record_tabular('Q(s,a_sample)', Q_values.mean().cpu().data.numpy())
        logger.record_tabular('Q Average', self.Q_average.cpu().data.numpy())
        logger.record_tabular('Real Loss', real_loss.cpu().data.numpy())
        logger.record_tabular('Fake Loss', fake_loss.cpu().data.numpy())
        logger.record_tabular('Discriminator Loss', discriminator_loss.cpu().data.numpy())
        logger.record_dict(create_stats_ordered_dict('Current_Q1', current_Q1.cpu().data.numpy()))
        logger.record_tabular('Rollout Length', rollout_len)
        logger.record_dict(create_stats_ordered_dict('Model Reward Predictions', generated_rewards))
        logger.record_tabular('Model Rollout Terminations', np.mean(terminated))
        logger.record_tabular('Weights Mean', float(weights.mean().cpu().data.numpy()))

    def train_from_torch(self, iterations):
        n_real_data, n_generated_data, warm_start = self.prepare_epoch_update()
        epoch_critic_loss = []

        """
        Update policy on both real and generated data
        """

        for _ in range(iterations):
            """
            Generate synthetic data using dynamics model
            """
            if self._n_train_steps_total % self.rollout_generation_freq == 0:
                generated_rewards, terminated, rollout_len = self.model_based_rollout()

            """
            Critic Training
            """
            batch = self.sample_batch(n_real_data=n_real_data, n_generated_data=n_generated_data)

            target_Q, current_Q1, current_Q2, critic_loss = self.optimize_critic(batch=batch, warm_start=warm_start)

            epoch_critic_loss.append(critic_loss.detach().cpu().numpy().mean())

            """
            Actor and Discriminator Training
            """
            fake_samples, true_samples, weights = self.get_true_fake_samples()

            generator_loss, discriminator_loss, real_loss, fake_loss, additional_logging = self.get_generator_and_discriminator_loss(fake_samples=fake_samples, true_samples=true_samples, weights=weights)

            Q_values, policy_loss = self.optimize_actor(batch=batch, warm_start=warm_start, generator_loss=generator_loss)

            discriminator_loss = self.optimize_discriminator(discriminator_loss=discriminator_loss)

            self.moving_average_updates(Q_values=Q_values)

            self._n_train_steps_total += 1

        self.update_epoch_critic_loss_thres(epoch_critic_loss=epoch_critic_loss, iterations=iterations)

        self.logging_common(warm_start, n_real_data, n_generated_data, true_samples, target_Q, critic_loss, epoch_critic_loss, policy_loss, generator_loss, Q_values, real_loss, fake_loss, discriminator_loss, current_Q1, rollout_len, generated_rewards, terminated, weights)

        return additional_logging

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))

        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))

        torch.save(self.discriminator.state_dict(), '%s/%s_discriminator.pth' % (directory, filename))
        torch.save(self.discriminator_optimizer.state_dict(), '%s/%s_discriminator_optimizer.pth' % (directory, filename))

        torch.save(self.log_lagrange.cpu(), '%s/%s_log_lagrange.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        self.actor_target = copy.deepcopy(self.actor)

        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/%s_critic_optimizer.pth' % (directory, filename)))
        self.critic_target = copy.deepcopy(self.critic)

        self.discriminator.load_state_dict(torch.load('%s/%s_discriminator.pth' % (directory, filename)))
        self.discriminator_optimizer.load_state_dict(torch.load('%s/%s_discriminator_optimizer.pth' % (directory, filename)))

        self.log_lagrange = torch.load('%s/%s_log_lagrange.pth' % (directory, filename))

    @property
    def networks(self):
        return self.actor

    @property
    def num_train_steps(self):
        return self._n_train_steps_total

    @property
    def num_epochs(self):
        return self._n_epochs

    def get_snapshot(self):
        snapshot = dict(
            dynamics_model=self.dynamics_model,
            actor=self.actor,
            critic=self.critic,
            discriminator=self.discriminator,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            log_lagrange=self.log_lagrange
        )
        return snapshot


