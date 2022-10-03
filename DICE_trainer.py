import copy
import torch
from utils import utils
from collections import defaultdict

from networks import Critic


class DICETrainer(object):
    def __init__(
            self,
            actor,
            dr_model,
            replay_buffer,
            device,
            state_dim,
            action_dim,
            correction,
            discount=0.99,
            beta=0.005,
    ):
        self.actor = actor
        self.replay_buffer = replay_buffer
        self.device = device
        self.discount = discount
        self.beta = beta

        self.correction = correction

        if self.correction == "GenDICE":
            self.lr = 0.003     # https://openreview.net/pdf?id=HkxlcnVFwB, pp.23
        else:
            self.lr = 0.0001

        self.tau = dr_model
        self.tau_target = copy.deepcopy(self.tau)

        self.nu = Critic(state_dim, action_dim).to(self.device)
        self.nu_target = copy.deepcopy(self.nu)

        self.tau_optim = torch.optim.Adam(self.tau.parameters(), lr=self.lr)
        self.nu_optim = torch.optim.Adam(self.nu.parameters(), lr=0.0001)

        self.v = torch.tensor(1.0, requires_grad=True, device=self.device)
        self.v_optim = torch.optim.Adam([self.v], lr=0.0001)
        self.lam = 1.0

        self.training_stats = defaultdict(list)

        utils.print_banner(f"Initialize DICETrainer-{self.correction} !!! beta={self.beta}, lr={self.lr}, weight_decay={self.tau_optim.__dict__['param_groups'][0]['weight_decay']}")

    def train_OPE(self, iterations, batch_size=256, record_training_stats=False):

        for idx in range(iterations):
            batch = self.replay_buffer.random_batch(batch_size, device=self.device)
            init_state = self.replay_buffer.all_start(device=self.device)
            init_action = self.actor(init_state).detach()
            next_action = self.actor(batch['next_observations']).detach()

            tau = self.tau(batch['observations'], batch['actions'])
            nu = self.nu.q1(batch['observations'], batch['actions'])
            nu_next = self.nu.q1(batch['next_observations'], next_action)
            init_nu = self.nu.q1(init_state, init_action)

            with torch.no_grad():
                tau_target = self.tau_target(batch['observations'], batch['actions'])
                nu_target = self.nu_target.q1(batch['observations'], batch['actions'])
                nu_next_target = self.nu_target.q1(batch['next_observations'], next_action)
                init_nu_target = self.nu_target.q1(init_state, init_action)

            # loss
            if self.correction == 'GenDICE':
                # https://openreview.net/pdf?id=HkxlcnVFwB Eq. (14)
                tau_loss = (1 - self.discount) * init_nu_target.mean() + self.discount * (tau * nu_next_target).mean() - \
                           (tau * (nu_target + 0.25 * nu_target.pow(2))).mean() + \
                           self.lam * (self.v.detach() * (tau.mean() - 1.) - 0.5 * self.v.detach().pow(2))
                nu_loss = (1 - self.discount) * init_nu.mean() + self.discount * (tau_target * nu_next).mean() - \
                          (tau_target * (nu + 0.25 * nu.pow(2))).mean()
                nu_loss = - nu_loss
            elif self.correction == 'DualDICE':
                tau_loss = ((nu_target - self.discount * nu_next_target) * tau).mean() - (tau.pow(3).mul(1. / 3)).mean() \
                           - (1 - self.discount) * init_nu_target.mean()
                nu_loss = ((nu - self.discount * nu_next) * tau_target).mean() - (tau_target.pow(3).mul(1. / 3.)).mean() - \
                          (1 - self.discount) * init_nu.mean()
                tau_loss = - tau_loss
            else:
                raise NotImplementedError

            self.tau_optim.zero_grad()
            self.nu_optim.zero_grad()

            tau_loss.backward()
            nu_loss.backward()

            if idx == int(iterations) - 1:
                dr_grad_norm_unclipped = self.calculate_dr_grad_norm()

            torch.nn.utils.clip_grad_value_(self.tau.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_value_(self.nu.parameters(), clip_value=1.0)

            self.tau_optim.step()
            self.nu_optim.step()

            if self.correction != 'DualDICE':
                v_loss = - self.lam * (self.v * (tau.mean().detach() - 1.) - 0.5 * self.v.pow(2))

                self.v_optim.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_value_(self.v, clip_value=1.0)
                self.v_optim.step()

            # update
            self._hard_update(self.tau, self.tau_target)
            self._hard_update(self.nu, self.nu_target)

        if record_training_stats:
            # only record the training stat of the last mini-batch
            self.training_stats['tau_loss'].append(float(utils.get_numpy(tau_loss)))
            self.training_stats['nu_loss'].append(float(utils.get_numpy(nu_loss)))
            if self.correction != 'DualDICE':
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

    def _hard_update(self, source_net, target_net):
        for sp, tp in zip(source_net.parameters(), target_net.parameters()):
            tp.data.copy_(sp.data)
