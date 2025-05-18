import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .veri import VeRi
from .bases import BaseImageDataset
from .samplers import RandomIdentitySampler
from .preprocessing import RandomErasing


def make_dataloader(cfg):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = VeRi(root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids
    # Lấy số lượng camera và view từ dataset
    camera_num = 20
    view_num = 1
    train_set = ImageDataset(dataset.train, train_transforms)
    
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

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=num_workers, collate_fn=val_collate_fn
    )
    
    train_loader_normal = None  # Hoặc tạo một dataloader thứ hai nếu cần

    
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, camera_num, view_num


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, viewid = self.dataset[index]
        img = read_image(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, pid, camid, viewid, img_path


def read_image(img_path):
    """Read image from file path"""
    from PIL import Image
    got_img = False
    try:
        img = Image.open(img_path).convert('RGB')
        got_img = True
    except IOError:
        print(f"IOError occurred when reading image from {img_path}")
    
    if not got_img:
        raise IOError(f"Failed to read image from {img_path}")
    
    return img


def train_collate_fn(batch):
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor([int(v) if isinstance(v, str) else v for v in viewids], dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor([int(v) if isinstance(v, str) else v for v in viewids], dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids, img_paths