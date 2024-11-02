import torch.nn as nn
import torch
import os

onlyReLU = False
withoutNorm = False

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

def get_generator(num_channels, latent_dim, hidden_dim, device, arch="cnn"):
    model = Generator(num_channels, latent_dim, hidden_dim, arch).to(device)
    model.apply(weights_init)
    return model

def get_discriminator(num_channels, hidden_dim, device, arch="cnn"):
    model = Discriminator(num_channels, hidden_dim, arch).to(device)
    model.apply(weights_init)
    return model

class Generator(nn.Module):
    def __init__(self, num_channels, latent_dim, hidden_dim, arch="cnn"):
        super(Generator, self).__init__()
        self.arch = arch
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_dim = num_channels * 32 * 32  # For mlp output reshape

        if arch == "cnn":
            # TODO START
            self.decoder = nn.Sequential(
                # state size: (latent_dim) x 1 x 1
                nn.ConvTranspose2d(latent_dim, hidden_dim * 4, kernel_size=4, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(hidden_dim * 4) if not withoutNorm else nn.Identity(),
                nn.ReLU(False) if onlyReLU else nn.LeakyReLU(0.2, inplace=True),
                # state size: (hidden_dim*4) x 4 x 4
                nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(hidden_dim * 2) if not withoutNorm else nn.Identity(),
                nn.ReLU(False) if onlyReLU else nn.LeakyReLU(0.2, inplace=True),
                # state size: (hidden_dim*2) x 8 x 8
                nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(hidden_dim) if not withoutNorm else nn.Identity(),
                nn.ReLU(False) if onlyReLU else nn.LeakyReLU(0.2, inplace=True),
                # state size: (hidden_dim) x 16 x 16
                nn.ConvTranspose2d(hidden_dim, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh()
                # state size: (num_channels) x 32 x 32
            )
            # TODO END
        elif arch == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim * 4),
                nn.BatchNorm1d(hidden_dim * 4) if not withoutNorm else nn.Identity(),
                nn.ReLU() if onlyReLU else nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2) if not withoutNorm else nn.Identity(),
                nn.ReLU() if onlyReLU else nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if not withoutNorm else nn.Identity(),
                nn.ReLU() if onlyReLU else nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, self.output_dim),
                nn.Tanh()
            )

    def forward(self, z):
        z = z.to(next(self.parameters()).device)
        if self.arch == "cnn":
            return self.decoder(z)
        elif self.arch == "mlp":
            z = z.squeeze(-1).squeeze(-1)
            out = self.decoder(z)
            return out.view(-1, self.num_channels, 32, 32)

    def restore(self, ckpt_dir):
        try:
            if os.path.exists(os.path.join(ckpt_dir, 'generator.bin')):
                path = os.path.join(ckpt_dir, 'generator.bin')
            else:
                path = os.path.join(ckpt_dir, str(max(int(name) for name in os.listdir(ckpt_dir))), 'generator.bin')
        except:
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, ckpt_dir, global_step):
        os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
        path = os.path.join(ckpt_dir, str(global_step), 'generator.bin')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]
    
class Discriminator(nn.Module):
    def __init__(self, num_channels, hidden_dim, arch="cnn"):
        super(Discriminator, self).__init__()
        self.arch = arch
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.input_dim = num_channels * 32 * 32  # For mlp input reshape

        if arch == "cnn":
            self.clf = nn.Sequential(
                nn.Conv2d(num_channels, hidden_dim, 4, 2, 1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=True),
                nn.BatchNorm2d(hidden_dim * 2) if not withoutNorm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=True),
                nn.BatchNorm2d(hidden_dim * 4) if not withoutNorm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_dim * 4, 1, 4, 1, 0, bias=True),
                nn.Sigmoid()
            )
        elif arch == "mlp":
            self.clf = nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim * 4),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2) if not withoutNorm else nn.Identity(),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if not withoutNorm else nn.Identity(),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        if self.arch == "cnn":
            return self.clf(x).view(-1, 1).squeeze(1)
        elif self.arch == "mlp":
            x = x.view(x.size(0), -1)  # Flatten the input for mlp
            return self.clf(x).view(-1, 1).squeeze(1)

    def restore(self, ckpt_dir):
        try:
            if os.path.exists(os.path.join(ckpt_dir, 'discriminator.bin')):
                path = os.path.join(ckpt_dir, 'discriminator.bin')
            else:
                path = os.path.join(ckpt_dir, str(max(int(name) for name in os.listdir(ckpt_dir))), 'discriminator.bin')
        except:
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, ckpt_dir, global_step):
        os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
        path = os.path.join(ckpt_dir, str(global_step), 'discriminator.bin')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]