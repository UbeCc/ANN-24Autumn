from torchvision.utils import make_grid
from torchvision.utils import save_image

import time
import GAN
from trainer import Trainer
from dataset import Dataset
from tensorboardX import SummaryWriter

from pytorch_fid import fid_score

import torch
import torch.optim as optim
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--latent_dim', default=16, type=int)
    parser.add_argument('--generator_hidden_dim', default=16, type=int)
    parser.add_argument('--discriminator_hidden_dim', default=16, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_training_steps', default=5000, type=int)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--saving_steps', type=int, default=1000)
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--data_dir', default='../data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='results', type=str, help='The path of the checkpoint directory')
    parser.add_argument('--log_dir', default='./runs', type=str)
    
    parser.add_argument('--run_id', type=str, default='default')
    parser.add_argument('--tag', type=str, default='default')
    parser.add_argument('--interpolation', action='store_true')
    parser.add_argument('--arch', default='cnn', type=str)
    args = parser.parse_args()

    timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    config = '{}-ldim{}-bsize{}-steps{}-ghidden{}-dhidden{}-{}-{}'.format(
        args.arch,
        args.latent_dim,
        args.batch_size,
        args.num_training_steps,
        args.generator_hidden_dim,
        args.discriminator_hidden_dim,
        args.tag,
        timestamp
    )

    if not args.do_train:
        args.ckpt_dir = os.path.join(args.ckpt_dir, args.run_id)
        config = args.run_id
    else:
        args.ckpt_dir = os.path.join(args.ckpt_dir, config)
    args.log_dir = os.path.join(args.log_dir, config)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    num_channels = 1

    dataset = Dataset(args.batch_size, args.data_dir)
    netG = GAN.get_generator(num_channels, args.latent_dim, args.generator_hidden_dim, device, args.arch)
    netD = GAN.get_discriminator(num_channels, args.discriminator_hidden_dim, device, args.arch)
    tb_writer = SummaryWriter(args.log_dir)

    if args.do_train:
        optimG = optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
        optimD = optim.Adam(netD.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
        trainer = Trainer(device, netG, netD, optimG, optimD, dataset, args.ckpt_dir, tb_writer)
        trainer.train(args.num_training_steps, args.logging_steps, args.saving_steps)
                # 64, 1, 32, 32
    print("ckpt dir:", args.ckpt_dir)
    print("folders:", str(max(int(step) for step in os.listdir(args.ckpt_dir) if step.isdigit())))
    restore_ckpt_path = os.path.join(args.ckpt_dir, str(max(int(step) for step in os.listdir(args.ckpt_dir) if step.isdigit())))
    netG.restore(restore_ckpt_path)

    num_samples = 3000    
    os.makedirs(f"results/config", exist_ok=True)

    if args.interpolation:
        with torch.no_grad():
            imgs = None
            for i in range(20): # 20 interpolations
                points, left, right = 10, 0.0, 1.0
                lefthand = torch.randn(1, netG.latent_dim, 1, device=device)
                righthand = torch.randn(1, netG.latent_dim, 1, device=device)
                weights = torch.linspace(left, right, points, device=device)
                vecs = torch.lerp(lefthand, righthand, weights)
                vecs = vecs.transpose(1, 2).view(points, netG.latent_dim, 1, 1)

                if imgs is None:
                    imgs = netG.forward(vecs)
                else:
                    imgs = torch.cat((imgs, netG.forward(vecs)), 0)
            imgs = make_grid(imgs, nrow=points, pad_value=0) * 0.5 + 0.5
            save_image(imgs,"results/interpolation.png")
            exit()

    for i in range(5):
        fids = []
        real_dl = iter(dataset.training_loader)
        real_imgs = None
        while real_imgs is None or real_imgs.size(0) < num_samples:
            imgs = next(real_dl)
            if real_imgs is None:
                real_imgs = imgs[0]
            else:
                real_imgs = torch.cat((real_imgs, imgs[0]), 0)
        real_imgs = real_imgs[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5

        with torch.no_grad():
            samples = None
            while samples is None or samples.size(0) < num_samples:
                noise = torch.randn(args.batch_size, netG.latent_dim, device=device)
                if args.arch == "mlp":
                    imgs = netG(noise)
                else:
                    imgs = netG(noise.unsqueeze(2).unsqueeze(3))
                if samples is None:
                    samples = imgs
                else:
                    samples = torch.cat((samples, imgs), 0)

        samples = samples[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5
        samples = samples.cpu()

        imgs = make_grid(samples, nrow=10, pad_value=0) *0.5 + 0.5
        save_image(imgs,f"results/config/samples-{config}-{i}.png")
        
        fid = fid_score.calculate_fid_given_images(real_imgs, samples, args.batch_size, device)
        tb_writer.add_scalar('fid', fid)
        print("FID score: {:.3f}".format(fid), flush=True)
        fids.append(fid)
    
    with open(f"results/{config}/fids.txt", "w") as f:
        f.write(str(fids))