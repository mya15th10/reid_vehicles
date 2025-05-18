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
        
        # Khởi tạo các thuộc tính trước khi gọi super().__init__
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_lr_init = warmup_lr_init
        self.warmup_t = warmup_t  # Đảm bảo thuộc tính này được gán
        self.t_in_epochs = t_in_epochs
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_std = noise_std
        self.noise_seed = noise_seed
        
        # Lưu các tham số ban đầu để phòng khi cần
        self._warmup_t = warmup_t  # Backup
        
        # Gọi super().__init__ sau khi khởi tạo các thuộc tính
        super(CosineLRScheduler, self).__init__(optimizer, -1)
        
        # Lưu base values
        self.base_values = [group['lr'] for group in self.optimizer.param_groups]
        self.update_groups(self.base_values)
        
        # Khởi tạo scheduler nếu cần
        if initialize:
            self.step(0)

    def _get_lr(self, t):
        # Đảm bảo thuộc tính warmup_t tồn tại
        if not hasattr(self, 'warmup_t') or self.warmup_t is None:
            self.warmup_t = getattr(self, '_warmup_t', 0)
        
        # Đảm bảo thuộc tính base_values tồn tại
        if not hasattr(self, 'base_values') or self.base_values is None:
            self.base_values = [group['lr'] for group in self.optimizer.param_groups]
        
        # Đảm bảo thuộc tính warmup_lr_init tồn tại
        if not hasattr(self, 'warmup_lr_init'):
            self.warmup_lr_init = 0.0
        
        # Đảm bảo thuộc tính lr_min tồn tại
        if not hasattr(self, 'lr_min'):
            self.lr_min = 0.0
        
        # Đảm bảo thuộc tính cycle_limit tồn tại
        if not hasattr(self, 'cycle_limit'):
            self.cycle_limit = 0
        
        # Đảm bảo thuộc tính t_initial tồn tại
        if not hasattr(self, 't_initial'):
            self.t_initial = 100  # Giá trị mặc định
        
        # Tính toán learning rates
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * (b - self.warmup_lr_init) / self.warmup_t for b in self.base_values]
        else:
            t = t - self.warmup_t
            if self.cycle_limit == 0 or (self.cycle_limit > 0 and t < self.cycle_limit * self.t_initial):
                lrs = [self.lr_min + 0.5 * (b - self.lr_min) * (1 + math.cos(math.pi * t / self.t_initial)) for b in self.base_values]
            else:
                lrs = [self.lr_min for _ in self.base_values]
        return lrs

    def get_lr(self):
        # Đảm bảo self.last_epoch tồn tại và hợp lệ
        if not hasattr(self, 'last_epoch') or self.last_epoch < 0:
            self.last_epoch = 0
        return self._get_lr(self.last_epoch)

    def step(self, epoch=None):
        # Xử lý tham số epoch
        if epoch is None:
            epoch = self.last_epoch + 1
        
        # Đảm bảo thuộc tính t_in_epochs tồn tại
        if not hasattr(self, 't_in_epochs'):
            self.t_in_epochs = True
        
        # Cập nhật last_epoch
        if self.t_in_epochs:
            self.last_epoch = math.floor(epoch)
        else:
            self.last_epoch = self.last_epoch + 1
        
        # Cập nhật learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        
        return self.get_lr()

    def update_groups(self, values):
        # Cập nhật learning rates cho tất cả param groups
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value