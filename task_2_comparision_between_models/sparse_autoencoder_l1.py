

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

def now():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# Sparse Autoencoder with L1 penalty
class SparseAutoencoderL1(nn.Module):
    def __init__(self):
        super(SparseAutoencoderL1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def train_sparse_autoencoder_l1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = SparseAutoencoderL1().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    l1_lambda = 1e-5

    epochs = 5
    start = now()
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.view(imgs.size(0), -1).to(device)

            outputs, encoded = model(imgs)
            mse_loss = criterion(outputs, imgs)
            l1_loss = l1_lambda * torch.mean(torch.abs(encoded))
            loss = mse_loss + l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")
    end = now()
    print(f"Training finished in {end - start:.2f} seconds")

    # Final Test MSE
    model.eval()
    mse_total = 0.0
    count = 0
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.view(imgs.size(0), -1).to(device)
            outputs, _ = model(imgs)
            mse_total += criterion(outputs, imgs).item() * imgs.size(0)
            count += imgs.size(0)
    print(f"\nFinal Test MSE: {mse_total / count:.6f}")

if __name__ == "__main__":
    train_sparse_autoencoder_l1()
