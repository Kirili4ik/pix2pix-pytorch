import torch
import random

import os
from torch.utils.data import DataLoader
import torchvision
import torchvision
from torchvision import transforms
from PIL import Image



class MyDataset(torch.utils.data.Dataset):
    """Custom dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with train/val/test the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(os.listdir(path=self.root_dir))


    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(idx+1)+'.jpg')
        image = Image.open(img_name)

        image = torchvision.transforms.functional.to_tensor(image)

        width = int(image.size(-1) / 2)
        image, target = image[:, :, width:], image[:, :, :width]

        if self.transform:
            image, target = self.transform(image, target)

        sample = {'x': image, 'target': target}

        return sample



def my_transforms_tr(sample, target):
    # normalize for right pixel values; for inference use backward op: *0.5 + 0.5
    sample = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sample)
    target = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(target)

    x = random.randint(0, 30)
    y = random.randint(0, 30)
    flip = random.random() > 0.5

    # to PIL and back cause we are obliged to use old torch vesrsions!!!
    sample = transforms.Resize([286, 286], Image.BICUBIC)(transforms.ToPILImage()(sample))
    target = transforms.Resize([286, 286], Image.BICUBIC)(transforms.ToPILImage()(target))

    sample = transforms.ToTensor()(sample)
    target = transforms.ToTensor()(target)

    #sample = transforms.RandomCrop(256)(sample)
    sample = sample[:, x:256 + x, y:256 + y]
    target = target[:, x:256 + x, y:256 + y]

    if flip:
        actual_flip = transforms.RandomHorizontalFlip()
        sample = actual_flip(sample)
        target = actual_flip(target)

    return sample, target


def my_transforms_val(sample, target):
    sample = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sample)
    target = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(target)

    return sample, target


