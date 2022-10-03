import copy
import torch
from utils import utils
from collections import defaultdict
import math


class VPMTrainer(object):
    def __init__(
            self,
            actor,
            dr_model,
            replay_buffer,
            device,
            discount=0.99,
            beta=0.005,
    ):
        self.actor = actor
        self.replay_buffer = replay_buffer
        self.device = device
        self.discount = discount
        self.beta = beta

        self.lr = 0.0003

        self.tau = dr_model
        self.tau_target = copy.deepcopy(self.tau)

        self.tau_optim = torch.optim.Adam(self.tau.parameters(), lr=self.lr)
        self.alpha = 1.0

        self.v = torch.tensor(1.0, requires_grad=True, device=self.device)
        self.v_optim = torch.optim.Adam([self.v], lr=self.lr)
        self.lam = 1.0

        self.T = 200.
        self.M = 10.

        self.training_stats = defaultdict(list)

        utils.print_banner(f"Initialize VPMTrainer !!! beta={self.beta}, weight_decay={self.tau_optim.__dict__['param_groups'][0]['weight_decay']}")

    def train_OPE(self, iterations, batch_size=256, record_training_stats=False):

        for idx in range(int(self.T * self.M)):
            self._alpha_update(idx + 1)

            batch = self.replay_buffer.random_batch(batch_size, device=self.device)
            init_state = self.replay_buffer.all_start(device=self.device)
            init_action = self.actor(init_state).detach()
            next_action = self.actor(batch['next_observations']).detach()

            target = (1 - self.discount) * self.tau(init_state, init_action).mean()
            target += self.discount * (self.tau(batch['next_observations'], next_action) * self.tau_target(batch['observations'], batch['actions']).detach()).mean()

            # https://arxiv.org/pdf/2003.00722.pdf Equation (16) & (17)

            tau_x = self.tau(batch['observations'], batch['actions'])

            tau_loss = 0.5 * torch.square(tau_x).mean() \
                       - (1 - self.alpha) * (tau_x * self.tau_target(batch['observations'], batch['actions']).detach()).mean() \
                       - self.alpha * target + \
                       self.lam * (2 * self.v.detach() * (tau_x.mean() - 1) - self.v.detach() ** 2)

            self.tau_optim.zero_grad()
            tau_loss.backward()

            if idx == int(self.T * self.M) - 1:
                dr_grad_norm_unclipped = self.calculate_dr_grad_norm()

            self.tau_optim.step()

            # v loss
            v_loss = - self.lam * (2 * self.v * (self.tau(batch['observations'], batch['actions']).mean().detach() - 1) - self.v ** 2)

            self.v_optim.zero_grad()
            v_loss.backward()
            self.v_optim.step()

            if idx % int(self.M) == 0:
                self._hard_update(self.tau, self.tau_target)

        if record_training_stats:
            # only record the training stat of the last mini-batch
            self.training_stats['tau_loss'].append(float(utils.get_numpy(tau_loss)))
            self.training_stats['v_loss'].append(float(utils.get_numpy(v_loss)))
            for k, v in dr_grad_norm_unclipped.items():
                self.training_stats['dr_grad_norm_unclipped_Layer_%i' % k].append(v)
            for k, v in self.calcululate_dr_weight_norm().items():
                self.training_stats['dr_weight_norm_Layer_%i' % k].append(v)

    def get_weights(self, states, actions):
        with torch.no_grad():
            weights = self.tau(states, actions)
        return weights      # unnormalized weights

    def save(self, filename, directory):
        torch.save(self.tau.state_dict(), '%s/%s_density_ratio.pth' % (directory, filename))

    def load(self, filename, directory):
        self.tau.load_state_dict(torch.load('%s/%s_density_ratio.pth' % (directory, filename)))
        self.tau_target = copy.deepcopy(self.tau)

    def calculate_dr_grad_norm(self):
        total_norm = defaultdict(int)
        for idxx, p in enumerate(self.tau.parameters()):
            param_norm = float(utils.get_numpy(p.grad.detach().data.norm(2))) if p.grad is not None and p.requires_grad else 0.
            total_norm[idxx // 2] += param_norm

        return total_norm

    def calcululate_dr_weight_norm(self):
        total_norm = defaultdict(int)
        for idxx, p in enumerate(self.tau.parameters()):
            param_norm = float(utils.get_numpy(p.detach().data.norm(2)))
            total_norm[idxx // 2] += param_norm

        return total_norm

    def _alpha_update(self, total_iters):
        self.alpha = 1.0 / math.sqrt(float(total_iters))

    def _hard_update(self, source_net, target_net):
        for sp, tp in zip(source_net.parameters(), target_net.parameters()):
            tp.data.copy_(sp.data)
