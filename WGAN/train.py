import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Critic, Generator, initialize_weights

from utils import gradient_penalty

# Hyperparameter
import torch_directml
device = torch_directml.device()
# device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
# WEIGHT_CLIP = 0.01
LAMBDA_GP = 10

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], 
            [0.5 for _ in range(CHANNELS_IMG)]), # mean, std
    ]
)

# dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
dataset = datasets.ImageFolder(root='./celeb_dataset', transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Critic(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(critic)

# opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
# opt_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f'logs/real')
writer_fake = SummaryWriter(f'logs/fake')
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch, (real, _) in enumerate(loader):
        real = real.to(device)
        print(f"Train real: {real.shape}")
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE, NOISE_DIM, 1, 1)).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)

            gp = gradient_penalty(critic, real, fake, device=device)

            # Maximize (E_real - E_fake) = Minimize (-(E_real - E_fake))
            # loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP*gp)
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            # for p in critic.parameters():
            # # Gradient clipping Để giới hạn giá trị của gradient trong một giới hạn nào đó
            # # để tránh cho gradient quá lớn hoặc quá nhỏ ảnh hưởng đến hội tụ của mô hình.
            #     p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        ### Train Generator: min -E[critic(gen_fake)]
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch % 100 == 0:
            print(
                f'Epoch [{epoch}|{NUM_EPOCHS}] \ '
                f'Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}'
            )
            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image("Mnist Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Mnist Real Images", img_grid_real, global_step=step)

            step += 1