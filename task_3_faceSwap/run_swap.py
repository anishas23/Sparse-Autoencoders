import torch
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np

# =====================
# Model definitions (same as in training scripts)
# =====================
class AutoencoderA(torch.nn.Module):
    def __init__(self, input_dim=12288, hidden_dim=512, sparsity_weight=1e-3):
        super(AutoencoderA, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, input_dim),
            torch.nn.Sigmoid()
        )
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent

class DecoderOnly(torch.nn.Module):
    def __init__(self, hidden_dim=512, output_dim=12288):
        super(DecoderOnly, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Sigmoid()
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
# Load Face A
# =====================
face_A = Image.open('image_A.png').convert("RGB")
x_A = transform(face_A).unsqueeze(0)  # shape: [1, 12288]

# =====================
# Load trained models
# =====================
autoencoder_A = AutoencoderA()
decoder_B = DecoderOnly()

autoencoder_A.load_state_dict(torch.load("autoencoder_A.pth"))
decoder_B.load_state_dict(torch.load("decoder_B.pth"))

autoencoder_A.eval()
decoder_B.eval()

# =====================
# Face swap: encode A → decode with B’s decoder
# =====================
with torch.no_grad():
    latent_A = autoencoder_A.encoder(x_A)
    swapped_face = decoder_B(latent_A)

# =====================
# Convert back to image, display, and save
# =====================
swapped_face_img = swapped_face.view(3, 64, 64).detach().numpy().transpose(1, 2, 0)

# Display
plt.imshow(swapped_face_img)
plt.title("Face-swapped: Face B from A")
plt.axis("off")
plt.show()

# Save as PNG
swapped_face_img_uint8 = (swapped_face_img * 255).astype(np.uint8)
swapped_image = Image.fromarray(swapped_face_img_uint8)
swapped_image.save("swapped_face.png")
print("✅ Swapped face saved as swapped_face.png")
