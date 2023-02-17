import torch

from dataloader.coco_loader import AquariumDetection, get_albumentation, collate_fn


def build_loader(cfg, split):
    datasets = None
    if split == 'train':
        datasets = AquariumDetection(root=cfg.dataset_path, transforms=get_albumentation(True))

    else:
        datasets = AquariumDetection(root=cfg.dataset_path, split=split, transforms=get_albumentation(False))
    dataloaders = torch.utils.data.DataLoader(
        datasets, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        collate_fn=collate_fn
    )
    return dataloaders

