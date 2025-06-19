import torch
import numpy as np
from torch.utils.data import DataLoader
from .custom_dataset import CustomVehicleDataset
from .bases import BaseImageDataset
from .samplers import RandomIdentitySampler

def make_dataloader(cfg):
    
    num_workers = cfg.DATALOADER.NUM_WORKERS

    # Use CustomVehicleDataset (now feature-based)
    dataset = CustomVehicleDataset(root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids
    
    # Camera and view numbers for your dataset
    camera_num = 2  # 2 cameras
    view_num = 1    # 1 view per camera
    
    train_set = FeatureDataset(dataset.train)
    
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True,
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    val_set = FeatureDataset(dataset.query + dataset.gallery)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=num_workers, collate_fn=val_collate_fn
    )
    
    train_loader_normal = None  # Not used in this implementation

    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, camera_num, view_num


class FeatureDataset(torch.utils.data.Dataset):
    """Dataset for feature vectors instead of images"""
    
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        features, pid, camid, viewid = self.dataset[index]
        
        # Convert features to tensor
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        else:
            features = torch.FloatTensor(np.array(features))
        
        return features, pid, camid, viewid, f"feature_{index}"  # fake path for compatibility


def train_collate_fn(batch):
    features, pids, camids, viewids, _ = zip(*batch)
    
    # Stack feature vectors instead of images
    features = torch.stack(features, dim=0)  # [batch_size, feature_dim]
    
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor([int(v) if isinstance(v, str) else v for v in viewids], dtype=torch.int64)
    
    return features, pids, camids, viewids


def val_collate_fn(batch):
    features, pids, camids, viewids, paths = zip(*batch)
    
    # Stack feature vectors
    features = torch.stack(features, dim=0)
    
    viewids = torch.tensor([int(v) if isinstance(v, str) else v for v in viewids], dtype=torch.int64)
    
    return features, pids, camids, viewids, paths