import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosineLRScheduler(_LRScheduler):
    def __init__(self,
                 optimizer,
                 t_initial,
                 t_mul=1.,
                 lr_min=0.,
                 decay_rate=1.,
                 warmup_lr_init=0,
                 warmup_t=0,
                 cycle_limit=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True):
        super(CosineLRScheduler, self).__init__(optimizer, -1)

        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_lr_init = warmup_lr_init
        self.warmup_t = warmup_t
        self.t_in_epochs = t_in_epochs
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_std = noise_std
        self.noise_seed = noise_seed

        self.base_values = [group['lr'] for group in self.optimizer.param_groups]
        self.update_groups(self.base_values)

        self.t_in_epochs = t_in_epochs
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_std = noise_std
        self.noise_seed = noise_seed

        if initialize:
            self.step(0)

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * (b - self.warmup_lr_init) / self.warmup_t for b in self.base_values]
        else:
            t = t - self.warmup_t
            if self.cycle_limit == 0 or (self.cycle_limit > 0 and t < self.cycle_limit * self.t_initial):
                lrs = [self.lr_min + 0.5 * (b - self.lr_min) * (1 + math.cos(math.pi * t / self.t_initial)) for b in self.base_values]
            else:
                lrs = [self.lr_min for _ in self.base_values]
        return lrs

    def step(self, epoch):
        if self.t_in_epochs:
            self.last_epoch = math.floor(epoch)
        else:
            self.last_epoch = self.last_epoch + 1
        
        lrs = self._get_lr(self.last_epoch)
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
        
        return lrs

    def update_groups(self, values):
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value