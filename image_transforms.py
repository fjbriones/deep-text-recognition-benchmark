from torchvision.transforms import transforms
import numpy as np
import torch
from torch import nn

np.random.seed(0)

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size, channels=3):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1

        self.channels=channels

        self.blur_h = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=channels)
        self.blur_v = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=channels)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(self.channels, 1)

        self.blur_h.weight.data.copy_(x.view(self.channels, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(self.channels, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

def get_simclr_pipeline_transform(size, s=1, channels=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    #Original
    f_size = min(size)
    kernel_size = [int(0.1*s) if int(0.1*s)%2 else int(0.1*s)+1 for s in size]
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size, scale=(0.8,1.0)),
                                          # transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          GaussianBlur(kernel_size=int(0.1*f_size), channels=1),
                                          transforms.ToTensor()])

    return data_transforms

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        # x = np.array(x)
        return [self.base_transform(x) for i in range(self.n_views)]
