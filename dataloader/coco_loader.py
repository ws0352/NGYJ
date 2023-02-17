# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
import copy
import os
# %matplotlib inline
import warnings

import albumentations as A
import cv2
import matplotlib.pyplot as plt
# You can write up to 20GB to the current directory (/kaggle/working
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision import datasets

warnings.filterwarnings("ignore")
from torchvision.utils import draw_bounding_boxes

from albumentations.pytorch import ToTensorV2


def collate_fn(batch):
    return tuple(zip(*batch))


def get_albumentation(train):
    if train:
        transform = A.Compose([
            A.Resize(320, 320),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(320, 320),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform


class AquariumDetection(datasets.VisionDataset):
    def __init__(
            self,
            root: str,
            split="train",
            transform=None,
            target_transform=None,
            transforms=None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.split = split
        # print(split)
        self.coco = COCO(os.path.join(root, split, "_annotations.coco.json"))
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = cv2.imread(os.path.join(self.root, self.split, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = copy.deepcopy(self._load_target(id))

        boxes = [t['bbox'] + [t['category_id']] for t in target]
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)

        image = transformed['image']
        boxes = transformed['bboxes']
        new_boxes = []
        for box in boxes:
            xmin = box[0]
            ymin = box[1]
            xmax = xmin + box[2]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(new_boxes, dtype=torch.float32)

        targ = {}
        targ["boxes"] = boxes
        targ["labels"] = torch.tensor([t["category_id"] for t in target], dtype=torch.int64)
        targ["image_id"] = torch.tensor([t["image_id"] for t in target])
        targ["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        targ["iscrowd"] = torch.tensor([t["iscrowd"] for t in target], dtype=torch.int64)

        return image.div(255), targ, targ["image_id"]

    def __len__(self) -> int:
        return len(self.ids)


if __name__ == '__main__':

    dataset_path = "..\\data\\flower"
    coc = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
    categories = coc.cats
    n_classes = len(categories.keys())
    print(n_classes, categories)
    classes = []
    for i in categories.items():
        classes.append(i[1]["name"])

    train_dataset = AquariumDetection(root=dataset_path, transforms=get_albumentation(True))
    val_dataset = AquariumDetection(root=dataset_path, split="valid", transforms=get_albumentation(False))
    test_dataset = AquariumDetection(root=dataset_path, split="test", transforms=get_albumentation(False))
    sample = train_dataset[12]
    img_int = torch.tensor(sample[0] * 255, dtype=torch.uint8)
    plt.imshow(draw_bounding_boxes(img_int,
                                   sample[1]['boxes'],
                                   [classes[i] for i in sample[1]['labels']],
                                   width=4).permute(1, 2, 0)
               )
    plt.show()
