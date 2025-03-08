from datetime import datetime

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw
import numpy as np


def collate_cifar10_fn(data) -> tuple[torch.Tensor, torch.Tensor]:
    # BxHxWxC
    images = torch.zeros((0, 3, 32, 32), dtype=torch.float32)
    for image, _ in data:
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        images = torch.cat([images, image.unsqueeze(dim=0)])
    images /= 127.5
    images -= 1
    # labels
    labels = torch.tensor([label for _, label in data], dtype=torch.int64)
    return images, labels

def collate_mnist_fn(data: list[tuple[Image, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    # images
    images = torch.zeros((0, 1, 28, 28), dtype=torch.float32)  # BxCxHxW, where H=28, W=28, C=1 for mnist
    for image, _ in data:
        images = torch.cat([images, torch.tensor(np.asarray(image)).view(1, 1, 28, 28)])
    # zero pad to 32x32
    images = torch.nn.functional.pad(images, (2, 2, 2, 2), mode='constant', value=0)
    images /= 127.5
    images -= 1
    # labels
    labels = torch.tensor([label for _, label in data], dtype=torch.int64)
    return images, labels

def make_grid(images: torch.Tensor) -> torch.Tensor:
    # BxCxHxW -> 1xCxnHxnW
    batch_size = images.size(0)
    return torchvision.utils.make_grid(images, nrow=round(np.sqrt(batch_size))).unsqueeze(dim=0)

def plot_labels(labels: torch.Tensor, h: int = 32, w: int = 32) -> torch.Tensor:
    # (B,) -> Bx1xHxW image tensors
    batch_size = labels.size(0)
    image_data = np.zeros((batch_size, 1, h, w), dtype=np.uint8)
    for i in range(batch_size):
        img = Image.fromarray(image_data[i, 0, :, :])
        d = ImageDraw.Draw(img)
        d.text((2, 2), f"{labels[i].numpy()}", fill=255, align="center", font_size=28)
        image_data[i, 0, :, :] = np.asarray(img)
    return torch.tensor(image_data, dtype=torch.float32)


def main():
    dataset_name = "cifar10"
    n_channels = {
        "mnist": 1,
        "cifar10": 3,
    }
    collate_fn = {
        "mnist": collate_mnist_fn,
        "cifar10": collate_cifar10_fn,
    }
    datasets = {
        "mnist": torchvision.datasets.MNIST("./", download=True),
        "cifar10": torchvision.datasets.CIFAR10("./", download=True),
    }
    dataset = datasets[dataset_name]
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn[dataset_name])
    log_name = f"{dataset_name}-vit-{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"
    writer = SummaryWriter(log_dir=f"runs/{log_name}")
    n_epochs = 100

    i = 0
    for _ in range(n_epochs):
        for images, labels in dataloader:
            writer.add_images("train/images", make_grid(images), i)
            writer.add_images("train/labels", make_grid(plot_labels(labels)), i)
            i += 1
    writer.close()

if __name__ == "__main__":
    torch.manual_seed(123)
    main()
