# sparse_autoencoder_mnist.py
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --- Hyperparameters ---
batch_size = 128
lr = 1e-3
epochs = 5
hidden_dim = 128
rho = 0.05           # desired average activation (sparsity target)
beta = 3.0           # weight of sparsity penalty
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data ---
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# --- Model ---
class SparseAE(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden),
            nn.Sigmoid(),
        )
        self.dec = nn.Sequential(
            nn.Linear(hidden, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.enc(x)
        x_hat = self.dec(h)
        return x_hat, h

model = SparseAE(hidden_dim).to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)
bce = nn.BCELoss(reduction="mean")

def kl_divergence(rho, rho_hat, eps=1e-8):
    rho_hat = torch.clamp(rho_hat, eps, 1 - eps)
    term1 = rho * torch.log(rho / rho_hat)
    term2 = (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    return torch.sum(term1 + term2)

# --- Training ---
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader, start=1):
        x = x.to(device)
        x_hat, h = model(x)
        recon_loss = bce(x_hat, x.view(x.size(0), -1))

        rho_hat = torch.mean(h, dim=0)
        sparsity_loss = kl_divergence(torch.tensor(rho, device=device), rho_hat)

        loss = recon_loss + beta * sparsity_loss / h.size(1)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item() * x.size(0)

        # --- Show batch loss ---
        if batch_idx % 50 == 0:  # print every 50 batches
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch} complete: avg_loss={total_loss/len(train_loader.dataset):.4f}")

# --- Quick evaluation: show reconstructions ---
model.eval()
x, _ = next(iter(test_loader))
x = x.to(device)[:8]
with torch.no_grad():
    x_hat, _ = model(x)
x_hat = x_hat.view(-1, 1, 28, 28).cpu()

# Plot originals vs reconstructions
plt.figure(figsize=(8,4))
for i in range(8):
    plt.subplot(2,8,i+1); plt.imshow(x[i,0].cpu(), cmap="gray"); plt.axis("off")
    plt.subplot(2,8,8+i+1); plt.imshow(x_hat[i,0], cmap="gray"); plt.axis("off")
plt.suptitle("Top: originals  |  Bottom: reconstructions (Sparse AE)")
plt.tight_layout()

# Save output instead of just showing
plt.savefig("reconstructions.png")
print("✅ Saved reconstructed images to reconstructions.png")
