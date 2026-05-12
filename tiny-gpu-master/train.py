"""
Step 1: Train a small MLP on MNIST (runs on your laptop)
Network: 784 -> 128 -> 10
Saves weights to: outputs/mlp_mnist.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

os.makedirs("outputs", exist_ok=True)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def train():
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_data  = datasets.MNIST("data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=256)

    model   = MLP()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("Training MLP on MNIST ...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                correct += (model(x).argmax(1) == y).sum().item()
        acc = correct / len(test_data) * 100
        print(f"  Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}%")

    torch.save(model.state_dict(), "outputs/mlp_mnist.pth")
    print("\nSaved: outputs/mlp_mnist.pth")
    return model


if __name__ == "__main__":
    train()
