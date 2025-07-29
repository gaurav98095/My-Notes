import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os

# ----- Hyperparameters ----- #
batch_size = 128
z_dim = 100
image_size = 28
channels = 1
epochs = 50
lr = 0.0002
beta1 = 0.5
num_classes = 10
embedding_dim = 50

# ----- Setup ----- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("generated_images", exist_ok=True)

# ----- Data ----- #
transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ----- Generator ----- #
class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, embedding_dim):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim + embedding_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        label_embedding = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([z, label_embedding], dim=1)
        return self.net(x)


# ----- Discriminator ----- #
class Discriminator(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        label_embedding = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        label_map = label_embedding.expand(-1, 1, 28, 28)
        x = torch.cat([img, label_map], dim=1)
        return self.net(x)


# ----- Model, Loss, Optimizer ----- #
generator = Generator(z_dim, num_classes, embedding_dim).to(device)
discriminator = Discriminator(num_classes, embedding_dim).to(device)
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# ----- Training ----- #
k = 1  # Generator updates per batch
p = 1  # Discriminator updates per batch

for epoch in range(1, epochs + 1):
    for i, (real_imgs, labels) in enumerate(train_loader):
        batch_size_curr = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        labels = labels.to(device)

        real = torch.ones(batch_size_curr, 1, device=device)
        fake = torch.zeros(batch_size_curr, 1, device=device)

        # Train Discriminator
        for _ in range(p):
            z = torch.randn(batch_size_curr, z_dim, 1, 1, device=device)
            fake_imgs = generator(z, labels)

            real_validity = discriminator(real_imgs, labels)
            d_real_loss = criterion(real_validity, real)

            fake_validity = discriminator(fake_imgs.detach(), labels)
            d_fake_loss = criterion(fake_validity, fake)

            d_loss = d_real_loss + d_fake_loss
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

        # Train Generator
        for _ in range(k):
            z = torch.randn(batch_size_curr, z_dim, 1, 1, device=device)
            fake_imgs = generator(z, labels)
            validity = discriminator(fake_imgs, labels)
            g_loss = criterion(validity, real)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        if i % 200 == 0:
            print(
                f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_loader)}] "
                f"D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}"
            )

    # Save sample images
    generator.eval()
    with torch.no_grad():
        z = torch.randn(64, z_dim, 1, 1, device=device)
        sample_labels = torch.arange(0, 10).repeat(7).to(device)[:64]
        samples = generator(z, sample_labels)
        samples = samples * 0.5 + 0.5
        save_image(samples, f"generated_images/epoch_{epoch}.png", nrow=8)
    generator.train()
