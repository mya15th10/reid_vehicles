import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, RandomIdentitySampler_DDP
from .custom_vehicle_dataset import CustomVehicleDataset  # Import your custom dataset

__factory = {
    'custom_vehicle': CustomVehicleDataset,  # Add your dataset here
}

def train_collate_fn(batch):
    """
    Collate function for training dataloader
    """
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids

def val_collate_fn(batch):
    """
    Collate function for validation dataloader
    """
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids, img_paths

def make_dataloader(cfg):
    """
    Create dataloaders for training and testing
    """
    
    # Data augmentation and preprocessing
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, 
                     mode='pixel', 
                     max_count=1, 
                     device='cpu'),
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    # Create dataset
    dataset_name = cfg.DATASETS.NAMES[0]  # Should be 'custom_vehicle'
    
    if dataset_name not in __factory:
        raise KeyError(f"Unknown dataset: {dataset_name}")
    
    dataset = __factory[dataset_name](root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    # Training dataset
    train_set = ImageDataset(dataset.train, train_transforms)
    
    # Use RandomIdentitySampler for training
    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            train_sampler = RandomIdentitySampler_DDP(
                dataset.train, 
                cfg.SOLVER.IMS_PER_BATCH, 
                cfg.DATALOADER.NUM_INSTANCE
            )
        else:
            train_sampler = RandomIdentitySampler(
                dataset.train, 
                cfg.SOLVER.IMS_PER_BATCH, 
                cfg.DATALOADER.NUM_INSTANCE
            )
    elif cfg.MODEL.DIST_TRAIN:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=train_sampler, 
        num_workers=num_workers,
        collate_fn=train_collate_fn, 
        drop_last=True
    )

    # Validation datasets
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, 
        batch_size=cfg.TEST.IMS_PER_BATCH, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader, len(dataset.query), num_classes, cam_num, view_num