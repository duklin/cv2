import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class ShallowModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1)
        self.conv_2 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=3, stride=1
        )
        self.linear = nn.Linear(in_features=20 * 12 * 12, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 20 * 12 * 12)
        x = self.linear(x)
        return x


class DeeperModel(nn.Module):
    def __init__(self, batch_norm: bool) -> None:
        super().__init__()
        self.batch_norm = batch_norm
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1)
        self.bn_1 = nn.BatchNorm2d(10)
        self.conv_2 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=3, stride=1
        )
        self.bn_2 = nn.BatchNorm2d(20)
        self.conv_3 = nn.Conv2d(
            in_channels=20, out_channels=40, kernel_size=3, stride=1
        )
        self.bn_3 = nn.BatchNorm2d(40)
        self.conv_4 = nn.Conv2d(
            in_channels=40, out_channels=80, kernel_size=3, stride=1
        )
        self.bn_4 = nn.BatchNorm2d(80)
        self.linear_1 = nn.Linear(in_features=80 * 4 * 4, out_features=200)
        self.linear_2 = nn.Linear(in_features=200, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = F.relu(self.bn_1(x) if self.batch_norm else x)
        x = self.conv_2(x)
        x = F.relu(self.bn_2(x) if self.batch_norm else x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv_3(x)
        x = F.relu(self.bn_3(x) if self.batch_norm else x)
        x = self.conv_4(x)
        x = F.relu(self.bn_4(x) if self.batch_norm else x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 80 * 4 * 4)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x


def train(
    num_epochs: int, train_loader: DataLoader, test_loader: DataLoader, model: nn.Module
) -> Tuple[List[float], List[float]]:
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.CrossEntropyLoss()

    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(images)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                if batch_idx % 50 == 0:
                    print(
                        f"Epoch : {epoch} [{batch_idx*len(images)}/{len(train_loader.dataset)} ({100.*batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.data.item():.6f}"
                    )

        model.eval()
        total_loss = 0
        correct = 0
        for images, labels in test_loader:
            output = model(images)
            loss = loss_func(output, labels)
            predicted = torch.max(output.data, 1)[1]
            correct += (predicted == labels).sum().item()
            total_loss += loss.data.item()
        test_losses.append(total_loss)
        test_accuracies.append(correct / len(test_loader.dataset))

    return test_losses, test_accuracies


def main():
    # Get data
    mnist_trainset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    mnist_testset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    # Training Parameters
    nepochs = 10
    batch_size = 64

    # Create Dataloaders
    train_loader = DataLoader(mnist_trainset, batch_size=batch_size)
    test_loader = DataLoader(mnist_testset, batch_size=batch_size)

    # Create Model
    shallow_model = ShallowModel()
    deep_model_no_bn = DeeperModel(batch_norm=False)
    deep_model_bn = DeeperModel(batch_norm=True)

    iter_models = lambda: zip(
        [shallow_model, deep_model_no_bn, deep_model_bn],
        ["shallow_model", "deep_model_no_bn", "deep_model_bn"],
    )

    # Train Model
    metrics = {"loss": {}, "accuracy": {}}

    for model, model_name in iter_models():
        losses, accuracies = train(nepochs, train_loader, test_loader, model)
        metrics["loss"][model_name] = losses
        metrics["accuracy"][model_name] = accuracies
        torch.save(model.state_dict(), os.path.join("models", f"{model_name}.pt"))

    # Save metrics
    with open("metrics.json", "w") as fout:
        json.dump(metrics, fout)

    # Plot metrics
    x_ticks = np.arange(nepochs)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    for axis, metric in zip([ax1, ax2], ["loss", "accuracy"]):
        axis.set_xlabel("Epoch")
        axis.grid(True)
        axis.set_xticks(x_ticks)
        for _, model_name in iter_models():
            axis.plot(metrics[metric][model_name], label=model_name)
        axis.legend()
    fig.savefig("metrics.png")


if __name__ == "__main__":
    main()
