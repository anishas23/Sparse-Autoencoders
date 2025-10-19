import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

# =====================
# Model definitions
# =====================
class AutoencoderA(nn.Module):
    def __init__(self, input_dim=12288, hidden_dim=512, sparsity_weight=1e-3):
        super(AutoencoderA, self).__init__()
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
        return self.sparsity_weight * torch.mean(torch.abs(latent))

class DecoderOnly(nn.Module):
    def __init__(self, hidden_dim=512, output_dim=12288):
        super(DecoderOnly, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(x)

# =====================
# Image transform
# =====================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# =====================
# Load Face B
# =====================
face_B = Image.open('image_B.png').convert("RGB")
x_B = transform(face_B).unsqueeze(0)  # shape: [1, 12288]

# =====================
# Load trained encoder from Face A
# =====================
autoencoder_A = AutoencoderA()
autoencoder_A.load_state_dict(torch.load("autoencoder_A.pth"))

# Freeze encoder parameters
for param in autoencoder_A.encoder.parameters():
    param.requires_grad = False

# =====================
# Initialize Decoder B
# =====================
decoder_B = DecoderOnly(hidden_dim=512, output_dim=12288)

# =====================
# Training setup
# =====================
criterion = nn.MSELoss()
optimizer = optim.Adam(decoder_B.parameters(), lr=1e-3)
epochs = 5000

# =====================
# Train Decoder B
# =====================
for epoch in range(epochs):
    optimizer.zero_grad()
    latent_A = autoencoder_A.encoder(x_B)
    output_B = decoder_B(latent_A)
    loss = criterion(output_B, x_B)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# =====================
# Save trained decoder
# =====================
torch.save(decoder_B.state_dict(), "decoder_B.pth")
print("✅ Training completed for Decoder B and saved as decoder_B.pth!")

