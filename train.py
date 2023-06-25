import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
# used for create parameters
import argparse
from dataset import Downloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on {device}')

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='MNIST', help='dataset to use')
args = argparser.parse_args()

LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_D = 64
FEATURES_G = 64
CHANNELS_IMG = 1

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)


if args.dataset == 'MNIST':
    dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
    CHANNELS_IMG = 1
elif args.dataset == 'CelebA':

    Downloader.download_celeb_a('celeb_dataset/')
    dataset = datasets.ImageFolder(root="celeb_dataset/", transform=transforms)
    CHANNELS_IMG = 3
else:
    raise Exception('dataset not supported')




dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
G = Generator(Z_DIM, CHANNELS_IMG, FEATURES_G).to(device)
D = Discriminator(CHANNELS_IMG, FEATURES_D).to(device)
initialize_weights(G)
initialize_weights(D)

optim_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optim_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

G.train()
D.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = G(noise)

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        D_real = D(real).reshape(-1)
        loss_D_real = criterion(D_real, torch.ones_like(D_real))
        D_fake = D(fake).reshape(-1)
        loss_D_fake = criterion(D_fake, torch.zeros_like(D_fake))
        loss_D = (loss_D_real + loss_D_fake) / 2

        D.zero_grad()
        loss_D.backward(retain_graph=True)
        optim_D.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = D(fake).reshape(-1)
        loss_G = criterion(output, torch.ones_like(output))
        G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if batch_idx % 5 == 0:
            print(
                f"\rEpoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_D:.4f}, loss G: {loss_G:.4f}",
                end=""
            )
        if batch_idx % 100 == 0:
            with torch.no_grad():
                fake = G(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image(
                    "Real Images", img_grid_real, global_step=step
                )
                writer_fake.add_image(
                    "Fake Images", img_grid_fake, global_step=step
                )
            step += 1