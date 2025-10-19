import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

# Define Sparse Autoencoder
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=12288, hidden_dim=512, sparsity_weight=1e-3):  # updated input_dim
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent

    def sparse_loss(self, latent):
        # L1 sparsity penalty
        return self.sparsity_weight * torch.mean(torch.abs(latent))


# Image transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten to 1D vector
])

# Load Face A
face_A = Image.open('image_A.png').convert("RGB")  # Replace with your actual image file
x_A = transform(face_A).unsqueeze(0)  # shape: [1, 12288]

# Initialize autoencoder with correct input size
autoencoder_A = SparseAutoencoder(input_dim=12288)
optimizer = optim.Adam(autoencoder_A.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Train autoencoder on Face A
epochs = 5000  # adjust if needed
for epoch in range(epochs):
    optimizer.zero_grad()
    output, latent = autoencoder_A(x_A)
    loss = criterion(output, x_A) + autoencoder_A.sparse_loss(latent)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

print("✅ Training completed for Face A!")
# Save trained model
torch.save(autoencoder_A.state_dict(), "autoencoder_A.pth")
print("✅ Saved autoencoder_A.pth")
