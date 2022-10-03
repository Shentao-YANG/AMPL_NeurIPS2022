import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import utils
from collections import defaultdict


class TdDiceRewardTrainer(object):
    def __init__(
            self,
            actor,
            dynamics_model,
            dr_model,
            replay_buffer,
            device,
            discount=0.99,
            beta=0.005,
            training_penalty="False",
            loss_type="l2",
            weight_decay=-1.
    ):
        self.actor = actor
        self.dynamics_model = dynamics_model
        self.replay_buffer = replay_buffer
        self.device = device
        self.discount = discount
        self.beta = beta

        self.dr = dr_model
        self.dr_target = copy.deepcopy(self.dr)
        self.training_penalty = training_penalty == "True"

        if not self.training_penalty:
            # Use usual optimizer
            if weight_decay < 0.:
                self.dr_optimizer = torch.optim.Adam(self.dr.parameters(), lr=1e-6)
            else:
                self.dr_optimizer = torch.optim.AdamW(self.dr.parameters(), lr=1e-6, weight_decay=weight_decay)
        else:
            # Use constraint optimizer
            if weight_decay < 0.:
                self.dr_optimizer = Constraint(self.dr.parameters(), torch.optim.Adam, lr=1e-6)
            else:
                self.dr_optimizer = Constraint(self.dr.parameters(), torch.optim.AdamW, lr=1e-6, weight_decay=weight_decay)
            self.dr_optimizer.g_constraint = 10.0
            utils.print_banner(f"Using penalized training in TdDiceRewardTrainer !!! With g_constraint={self.dr_optimizer.g_constraint}")

        self.training_stats = defaultdict(list)

        if loss_type == "l2":
            self.loss = nn.MSELoss()
        elif loss_type == "l1":
            self.loss = nn.L1Loss()
        elif loss_type == "Huber":
            self.loss = nn.HuberLoss(delta=1.0)
        else:
            raise NotImplementedError(f"Receive loss_type={loss_type}, should be ('l2', 'l1', 'Huber').")

        utils.print_banner(f"Initialize TdDiceRewardTrainer !!! training_penalty={self.training_penalty}, beta={self.beta}, loss_type={self.loss}, weight_decay={self.dr_optimizer.__dict__['param_groups'][0]['weight_decay']}")

    def get_reward(self, states, actions):
        fake_transitions = self.dynamics_model.sample(torch.cat([states, actions], dim=-1))
        if (fake_transitions != fake_transitions).any():
            fake_transitions[fake_transitions != fake_transitions] = 0
        fake_rewards = fake_transitions[:, :1]

        return fake_rewards

    def train_OPE(self, iterations, batch_size=256, record_training_stats=False):

        for idx in range((int(iterations))):

            batch = self.replay_buffer.random_batch(batch_size, device=self.device)
            start_state = self.replay_buffer.all_start(device=self.device)

            with torch.no_grad():
                start_action = self.actor(start_state)
                init_Q = self.get_reward(start_state, start_action)
                start_Q = (1 - self.discount) * init_Q.mean()

                next_action = self.actor(batch['next_observations'])
                w_sa_target = self.dr_target(batch['observations'], batch['actions'])
                target = start_Q + self.discount * torch.mean((1. - batch['terminals']) * self.get_reward(batch['next_observations'], next_action) * w_sa_target)

            if self.training_penalty:
                self.dr_optimizer.zero_grad()
                # Get the gradient for the constraint function
                lamba_term = self.dr(batch['observations'], batch['actions']).mean()
                lamba_term.backward()
                self.dr_optimizer.g_value = lamba_term.item()
                self.dr_optimizer.first_step(zero_grad=True)
                # Get the gradient for the training objective function
                w_sa = self.dr(batch['observations'], batch['actions'])
                dr_loss = self.loss((w_sa * self.get_reward(batch['observations'], batch['actions'])).mean(), target)
                dr_loss.backward()
                # Gradient correction
                self.dr_optimizer.second_step()

                if idx == int(iterations) - 1:
                    dr_grad_norm_unclipped = self.calculate_dr_grad_norm()

                torch.nn.utils.clip_grad_norm_(self.dr.parameters(), max_norm=1., error_if_nonfinite=True)
                self.dr_optimizer.base_optimizer.step()
            else:
                w_sa = self.dr(batch['observations'], batch['actions'])
                dr_loss = self.loss((w_sa * self.get_reward(batch['observations'], batch['actions'])).mean(), target)
                self.dr_optimizer.zero_grad()
                dr_loss.backward()

                if idx == int(iterations) - 1:
                    dr_grad_norm_unclipped = self.calculate_dr_grad_norm()

                torch.nn.utils.clip_grad_norm_(self.dr.parameters(), max_norm=1., error_if_nonfinite=True)
                self.dr_optimizer.step()

            # moving average target network updates
            for param, target_param in zip(self.dr.parameters(), self.dr_target.parameters()):
                target_param.data.copy_(self.beta * param.data + (1. - self.beta) * target_param.data)

        if record_training_stats:
            # only record the training stat of the last mini-batch
            self.training_stats['dr_loss'].append(float(utils.get_numpy(dr_loss)))
            if self.training_penalty:
                self.training_stats['lambda_term'].append(float(utils.get_numpy(lamba_term)))
            for k, v in dr_grad_norm_unclipped.items():
                self.training_stats['dr_grad_norm_unclipped_Layer_%i' % k].append(v)
            for k, v in self.calcululate_dr_weight_norm().items():
                self.training_stats['dr_weight_norm_Layer_%i' % k].append(v)

    def get_weights(self, states, actions):
        with torch.no_grad():
            weights = self.dr(states, actions)
        return weights      # unnormalized weights

    def save(self, filename, directory):
        torch.save(self.dr.state_dict(), '%s/%s_density_ratio.pth' % (directory, filename))

    def load(self, filename, directory):
        self.dr.load_state_dict(torch.load('%s/%s_density_ratio.pth' % (directory, filename)))
        self.dr_target = copy.deepcopy(self.dr)

    def calculate_dr_grad_norm(self):
        total_norm = defaultdict(int)
        for idxx, p in enumerate(self.dr.parameters()):
            param_norm = float(utils.get_numpy(p.grad.detach().data.norm(2))) if p.grad is not None and p.requires_grad else 0.
            total_norm[idxx // 2] += param_norm

        return total_norm

    def calcululate_dr_weight_norm(self):
        total_norm = defaultdict(int)
        for idxx, p in enumerate(self.dr.parameters()):
            param_norm = float(utils.get_numpy(p.detach().data.norm(2)))
            total_norm[idxx // 2] += param_norm

        return total_norm


class Constraint(torch.optim.Optimizer):
    """
    first_step: gradient of objective 1, and log the grad,
    second_step: gradient of objective 2, and do something based on the logged gradient at step one
    closure: the objective 2 for second step
    """

    def __init__(self, params, base_optimizer, g_star=0.05, alpha=1, beta=1, **kwargs):
        defaults = dict(g_star=g_star, **kwargs)
        super(Constraint, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.g_star = g_star
        self.alpha = alpha
        self.beta = beta
        self.g_constraint = 0.
        self.g_value = torch.tensor([1.]).item()

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                constraint_grad = torch.ones_like(p.grad) * p.grad  # deepcopy, otherwise the c_grad would be a pointer
                self.state[p]["constraint_grad"] = constraint_grad

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        '''
        calculate the projection here
        '''
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                phi_x = min(self.alpha * (self.g_value - self.g_constraint), self.beta * torch.norm(self.state[p]["constraint_grad"]) ** 2)
                adaptive_step_x = F.relu((phi_x - (p.grad * self.state[p]["constraint_grad"]).sum()) / (1e-8 + self.state[p]["constraint_grad"].norm().pow(2)))
                p.grad.add_(adaptive_step_x * self.state[p]["constraint_grad"])

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, g_value=None, g_constraint=None):
        assert closure is not None, "Requires closure, but it was not provided, raise an error"
        assert g_value is not None, "Requires g value"
        assert g_constraint is not None, "Requires g constraint"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.g_value = g_value
        self.g_constraint = g_constraint
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
