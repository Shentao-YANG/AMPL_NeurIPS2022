import abc
from collections import OrderedDict
from utils.utils import np_to_pytorch_batch, get_numpy, print_banner
from utils.logger import logger
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict


class TorchTrainer(object, metaclass=abc.ABCMeta):
    def __init__(self, device):
        self._num_train_steps = 0
        self.device = device

    def train(self, np_batch):
        self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch, device=self.device)
        self.train_from_torch(batch)

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_steps),
        ])

    def train_from_torch(self, batch):
        pass

    @property
    def networks(self):
        pass


class ModelTrainer(TorchTrainer):
    def __init__(
            self,
            ensemble,
            device,
            replay_buffer,
            num_elites=None,
            learning_rate=1e-3,
            batch_size=256,
            optimizer_class=optim.Adam,
    ):
        super().__init__(device=device)

        self.ensemble = ensemble
        self.ensemble_size = ensemble.ensemble_size
        self.num_elites = min(num_elites, self.ensemble_size) if num_elites else self.ensemble_size

        self.obs_dim = ensemble.obs_dim
        self.action_dim = ensemble.action_dim
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer
        self._state = {}
        self._snapshots = {i: (None, 1e32) for i in range(self.ensemble_size)}

        self.w_stats = defaultdict(list)

        self.optimizer = self.construct_optimizer(optimizer_class, learning_rate)

    def construct_optimizer(self, optimizer_class, lr):
        decays = [.000025, .00005, .000075, .000075, .0001]
        fcs = self.ensemble.fcs + [self.ensemble.last_fc]

        if self.ensemble.separate_mean_var:
            decays.append(.0001)
            fcs += [self.ensemble.last_fc_std]

        opt_params = [{'params': fcs[i].parameters(), 'weight_decay': decays[i]} for i in range(len(fcs))]
        # no weight_decay for the max/min_logstd
        opt_params.extend([{'params': self.ensemble.max_logstd, 'weight_decay': 0.}, {'params': self.ensemble.min_logstd, 'weight_decay': 0.}])

        optimizer = optimizer_class(opt_params, lr=lr)

        return optimizer

    def save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / abs(best)
            if improvement > 0.01:
                self._snapshots[i] = (epoch, np.mean(get_numpy(current)))
                self._state[i] = self.ensemble.get_idv_model_state(i)
                updated = True

        return updated

    def calculate_weights_from_dr(self, sa, dr):
        # states = sa[:, :self.obs_dim]
        # actions = sa[:, self.obs_dim:]
        w = torch.empty(0, 1, device=self.device)
        num_split = sa.shape[0] // 2048
        if self.replay_buffer.device == "numpy":
            sa_split = np.array_split(sa, indices_or_sections=num_split, axis=0)
            for i in range(len(sa_split)):
                w = torch.vstack((w, dr.get_weights(states=torch.from_numpy(sa_split[i][:, :self.obs_dim]).float().to(self.device),
                                                    actions=torch.from_numpy(sa_split[i][:, self.obs_dim:]).float().to(self.device))))
            w = w.cpu().numpy()
        else:
            sa_split = torch.split(sa, split_size_or_sections=num_split, dim=0)
            for i in range(len(sa_split)):
                w = torch.vstack((w, dr.get_weights(states=sa_split[i][:, :self.obs_dim], actions=sa_split[i][:, self.obs_dim:])))

        return w

    def record_w_states(self, weights, normalized=False):
        suffix = "normed" if normalized else "unnormed"
        if isinstance(weights, np.ndarray):
            self.w_stats[f"mean_{suffix}"].append(float(weights.mean()))
            self.w_stats[f"std_{suffix}"].append(float(weights.std()))
            self.w_stats[f"sum_{suffix}"].append(float(weights.sum()))
            self.w_stats[f"min_{suffix}"].append(float(weights.min()))
            self.w_stats[f"pct25_{suffix}"].append(float(np.quantile(weights, 0.25)))
            self.w_stats[f"median_{suffix}"].append(float(np.quantile(weights, 0.5)))
            self.w_stats[f"pct75_{suffix}"].append(float(np.quantile(weights, 0.75)))
            self.w_stats[f"max_{suffix}"].append(float(weights.max()))
        else:
            self.w_stats[f"mean_{suffix}"].append(float(get_numpy(weights.mean())))
            self.w_stats[f"std_{suffix}"].append(float(get_numpy(weights.std())))
            self.w_stats[f"sum_{suffix}"].append(float(get_numpy(weights.sum())))
            self.w_stats[f"min_{suffix}"].append(float(get_numpy(weights.min())))
            self.w_stats[f"pct25_{suffix}"].append(float(get_numpy(torch.quantile(weights, 0.25))))
            self.w_stats[f"median_{suffix}"].append(float(get_numpy(weights.median())))
            self.w_stats[f"pct75_{suffix}"].append(float(get_numpy(torch.quantile(weights, 0.75))))
            self.w_stats[f"max_{suffix}"].append(float(get_numpy(weights.max())))

    def train_from_buffer(self, holdout_pct=0.2, max_grad_steps=1000, epochs_since_last_update=5, dr_trainer=None, first_training=False):

        data = self.replay_buffer.get_transitions()
        x = data[:, :self.obs_dim + self.action_dim]    # inputs  s, a
        y = data[:, self.obs_dim + self.action_dim:]    # predict r, ns
        y[:, -self.obs_dim:] -= x[:, :self.obs_dim]     # predict delta in the state
        # calculate the unnormalized weight w(s,a), w: (x.shape[0], 1) and in the same device as input x
        if first_training:
            self.record_w_states(weights=self.calculate_weights_from_dr(sa=x, dr=dr_trainer), normalized=False)
            w = np.ones((x.shape[0], 1)) if self.replay_buffer.device == "numpy" else torch.ones((x.shape[0], 1), device=self.device)
        else:
            w = self.calculate_weights_from_dr(sa=x, dr=dr_trainer)

        self.record_w_states(weights=w, normalized=False)
        w = w / w.mean()        # normalized w so that w.sum() = w.shape[0]
        self.record_w_states(weights=w, normalized=True)

        if self.replay_buffer.store_na_w:
            self.replay_buffer.copy_weights(weights=w)

        # get normalization statistics
        # normalize the delta in y_test
        if self.replay_buffer.device == "numpy":
            self.ensemble.fit_input_stats(data=x, y=y)
            y[..., self.ensemble.rns_np] = (y[..., self.ensemble.rns_np] - self.ensemble.delta_obs_mu.data.cpu().numpy()) / self.ensemble.delta_obs_std.data.cpu().numpy()
        else:
            self.ensemble.fit_input_stats_torch(data=x, y=y)
            y[..., self.ensemble.rns_torch] = (y[..., self.ensemble.rns_torch] - self.ensemble.delta_obs_mu.data) / self.ensemble.delta_obs_std.data

        # generate holdout set
        inds = np.random.permutation(data.shape[0]) if self.replay_buffer.device == "numpy" else torch.randperm(data.shape[0], device=self.device)

        n_train = max(int((1. - holdout_pct) * data.shape[0]), data.shape[0] - 8092)
        # n_test = data.shape[0] - n_train
        x_train, y_train, w_train = x[inds[:n_train]], y[inds[:n_train]], w[inds[:n_train]]
        x_test, y_test, w_test = x[inds[n_train:]], y[inds[n_train:]], w[inds[n_train:]]
        if self.replay_buffer.device == "numpy":
            x_test, y_test, w_test = torch.from_numpy(x_test).float().to(self.device), torch.from_numpy(y_test).float().to(self.device), torch.from_numpy(w_test).float().to(self.device)
        w_test = w_test.repeat(self.ensemble_size, 1).reshape(self.ensemble_size, -1)

        # train until holdout set converge
        num_epochs, num_steps = 0, 0
        num_epochs_since_last_update = 0
        best_holdout_loss = float('inf')
        num_batches = int(1000)

        while num_epochs_since_last_update < epochs_since_last_update and num_steps < max_grad_steps:
            # generate idx for each model to bootstrap
            self.ensemble.train()
            for b in range(num_batches):
                b_idxs = np.random.randint(n_train, size=(self.ensemble_size * self.batch_size)) if self.replay_buffer.device == "numpy" else torch.randint(high=n_train, size=(self.ensemble_size * self.batch_size,), device=self.device)
                x_batch, y_batch, w_batch = x_train[b_idxs], y_train[b_idxs], w_train[b_idxs]
                if self.replay_buffer.device == "numpy":
                    x_batch, y_batch, w_batch = torch.from_numpy(x_batch).float().to(self.device), torch.from_numpy(y_batch).float().to(self.device), torch.from_numpy(w_batch).float().to(self.device)
                x_batch = x_batch.view(self.ensemble_size, self.batch_size, -1)
                y_batch = y_batch.view(self.ensemble_size, self.batch_size, -1)
                w_batch = w_batch.view(self.ensemble_size, self.batch_size)
                loss = self.ensemble.get_loss(x_batch, y_batch, w_batch, first_training=first_training)
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

            # stop training based on holdout loss improvement
            self.ensemble.eval()
            with torch.no_grad():
                holdout_errors, holdout_losses = self.ensemble.get_loss(x_test, y_test, w_test, split_by_model=True, return_l2_error=True, first_training=first_training)    # (losses, l2_errors)

            updated = self.save_best(epoch=num_epochs + 1, holdout_losses=holdout_losses)
            if num_epochs == 0 or updated:
                num_epochs_since_last_update = 0
            else:
                num_epochs_since_last_update += 1

            holdout_loss = sum(sorted(holdout_losses)[:self.num_elites]) / self.num_elites

            num_steps += num_batches
            num_epochs += 1

            if num_epochs % 1 == 0:
                # Logging period: 1
                logger.record_tabular('Model Training Epochs', num_epochs)
                logger.record_tabular('Num epochs since last update', num_epochs_since_last_update)
                logger.record_tabular('Model Training Steps', num_steps)
                logger.record_tabular("Model Training Loss", loss.cpu().data.numpy())
                logger.record_tabular("Model Holdout Loss", np.mean(get_numpy(sum(holdout_losses))) / self.ensemble_size)
                logger.record_tabular('Model Elites Holdout Loss', np.mean(get_numpy(holdout_loss)))

                for i in range(self.ensemble_size):
                    name = 'Model%d' % i
                    logger.record_tabular(name + ' Loss', np.mean(get_numpy(holdout_losses[i])))
                    logger.record_tabular(name + ' Error', np.mean(get_numpy(holdout_errors[i])))
                    logger.record_tabular(name + ' Best', self._snapshots[i][1])
                    logger.record_tabular(name + ' Best Epoch', self._snapshots[i][0])

                logger.dump_tabular(with_timestamp=False)

        self.ensemble.load_model_state_from_dict(state_dict=self._state)
        with torch.no_grad():
            holdout_errors, holdout_losses = self.ensemble.get_loss(x_test, y_test, w_test, split_by_model=True, return_l2_error=True, first_training=first_training)  # (losses, l2_errors)
        for i in range(self.ensemble_size):
            print_banner(f"Model {i}: Loss {np.mean(get_numpy(holdout_losses[i])):.8f}; Error {np.mean(get_numpy(holdout_errors[i])):.8f}; From epoch {self._snapshots[i][0]}")

        print_banner(f"OLD ensemble elites are {self.ensemble.elites}")
        self.ensemble.elites = np.argsort([np.mean(get_numpy(x)) for x in holdout_losses])[:self.num_elites]
        print_banner(f"NEW ensemble elites are {self.ensemble.elites}, their holdout losses are {[np.mean(get_numpy(holdout_losses[_idx])) for _idx in self.ensemble.elites]}")
        print_banner(f"MAX_logstd: {self.ensemble.max_logstd.data.cpu().numpy()}")
        print_banner(f"MIN_logstd: {self.ensemble.min_logstd.data.cpu().numpy()}")
        for k, v in self.w_stats.items():
            print(f"{k}: {np.round(v, 2)}")
            print('-' * 30)

        self._state = {}
        self._snapshots = {i: (None, 1e32) for i in range(self.ensemble_size)}

    def train_from_torch(self, batch, idx=None):
        raise NotImplementedError

    @property
    def networks(self):
        return [
            self.ensemble
        ]

    def get_snapshot(self):
        return dict(
            ensemble=self.ensemble
        )
