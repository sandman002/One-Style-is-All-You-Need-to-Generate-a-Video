import argparse
import math
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import lpips
from model_projgen import Generator
import glob

from pathlib import Path
import PIL.Image
import numpy as np
import random
import torchvision

import cv2
from torchvision.utils import make_grid


def list_dirs(path):
    return [os.path.basename(x) for x in filter(
        os.path.isdir, glob.glob(os.path.join(path, '*')))]

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )



if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Video projector to the generator's temporal latent space"
    )

    parser.add_argument(
        "--ckpt", type=str, default='./ckpt/080000.pt', help="path to the model checkpoint"
    )
    #./ckpt_mead_ntip_rem/117000.pt
    parser.add_argument(
        "--size", type=int, default=128, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--genonly", action='store_true') #if you do not want to optimize for m_t vector and simply only recast a new person to previously optimized m_t

    parser.add_argument(
        "--nframes", type=int, default=20, help="num of frames to optimize"
    )
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--step", type=int, default=250, help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )


    parser.add_argument(
        "--re_person",
        type=int,
        default=19,
        help="Id of the person to be recasted to",
    )
    parser.add_argument(
        "--re_label",
        type=int,
        default=0,
        help="action label to be recasted to",
    )
    parser.add_argument(
        "--maindir", type=str, default='./projection/input/', help="root path containing folders of video frames to be inverted"
    )
    parser.add_argument(
        "--outputdir", type=str, default='./projection/output/', help="folder containing the optimized frames, latent vectors and reprojection results"
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=8,
        help="number of action classes in the training dataset",
    )

    parser.add_argument(
        "--num_person",
        type=int,
        default=30,
        help="num of person in the training dataset",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="num of frames to generate",
    )

    parser.add_argument(
        "--latent",
        type=int,
        default=128,
        help="total latent dimension",
    )

    parser.add_argument(
        "--dyn_size",
        type=int,
        default=64,
        help="total t2v components",
    )


    parser.add_argument("--mse", type=float, default=100, help="weight of the mse loss")

    args = parser.parse_args()

    n_mean_latent = 10000

    resize = min(args.size, 256)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    main_dir = args.maindir
    out_dir = args.outputdir

    os.makedirs(out_dir,exist_ok=True)
    directories = list_dirs(main_dir)
    print('{} dir found'.format(len(directories)))
    print(directories)

    args.channel_multiplier=2
    args.n_mlp=8

    g_ema = Generator(
    args.size, args.latent- args.dyn_size, args.num_person,
    args.num_classes, args.latent-args.dyn_size, 
    args.dyn_size, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    nframes=args.nframes

    for kk in range(len(directories)):
        dataroot = main_dir + '/' + directories[kk]
        flist = (Path(dataroot).glob('*.jpg'))
        files = [x for x in flist if x.is_file()]
        outdir = out_dir + '/' + directories[kk]
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if args.genonly == False:
            imgs = []   
            if nframes>len(files):
                nframes = len(files)
            print("Optimizing for ", nframes,' frames')
            for ii in range(0,nframes):
                img = transform(Image.open(files[ii]).convert("RGB"))
                imgs.append(img)
     
            strs = str(directories[kk]).split('_')
            print(strs)
            imgs = torch.stack(imgs, 0).to(device)

            pid = int(strs[0])
            lid = int(strs[1])

            with torch.no_grad():
                sample_gen_tau = torch.zeros(n_mean_latent,1).to(device).type(torch.cuda.FloatTensor)
                for jj in range(n_mean_latent):
                    sample_gen_tau[jj,0] = (torch.cuda.FloatTensor(torch.randint(0, 40, (1,1), device=device).type(torch.cuda.FloatTensor)))
                sample_dyn_noise = torch.randn(n_mean_latent, args.dyn_size, device=device)
         
                dynstyle = g_ema.dyn_sty(sample_dyn_noise) # to w space from z space with 4 layer mlps
      
                latent_out = dynstyle#torch.hstack([styles[0], dynstyle])

                latent_mean = latent_out.mean(0)
                latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
     
            percept = lpips.PerceptualLoss(
                model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
            )

            noises_single = g_ema.make_noise()
            noises = []
            for noise in noises_single:
                noises.append(noise.repeat(1, 1, 1, 1).normal_())

            latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(1, 1)
            
            latent_in.requires_grad = True

            for noise in noises:
                noise.requires_grad = True

            optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

            pbar = tqdm(range(args.step))

            pid = int(strs[0])
            lid = int(strs[1])
            person_list = [pid]
            sample_person = torch.LongTensor(person_list).to(device)
            sample_label = [lid]
            sample_label = torch.LongTensor(sample_label).to(device)
            z0 = torch.zeros(1,1).to(device).type(torch.cuda.FloatTensor)
            
            for i in pbar:
                t = i / args.step
                lr = get_lr(t, args.lr)
                optimizer.param_groups[0]["lr"] = lr
                noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
                latent_n = latent_noise(latent_in, noise_strength.item())

                imggens = []
                dist = []
                for ff in range(0,nframes):
                    img_gen, _ = g_ema(sample_person, latent_n, class_label=sample_label, transition_factor=1, timepoints=z0+ff, input_is_latent=True)
                    imggens.append(img_gen)
                   
                img_gen = torch.stack(imggens).squeeze(1)

                batch, channel, height, width = img_gen.shape

                Path(outdir+"/opti/").mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(
                            img_gen,
                            outdir+'/opti/' + str(i).zfill(4)+'.png',
                            nrow=int(5),
                            normalize=True,
                            range=(-1, 1),
                        )


                p_loss = percept(img_gen, imgs).sum()
                n_loss = noise_regularize(noises)
                mse_loss = F.mse_loss(img_gen, imgs)

                loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss # + 0.5*sum(dist)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                noise_normalize_(noises)
                pbar.set_description(
                    (
                        f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                        f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                    )
                )


            Path(outdir+"/pvec/").mkdir(parents=True, exist_ok=True)
            np.savez(outdir+'/pvec/' + str(0).zfill(4)+'.npz', w=latent_n.detach().cpu().numpy())

        lat = np.load(outdir+'/pvec/' + str(0).zfill(4)+'.npz')

        latn = lat['w']
        latn = torch.from_numpy(latn).to(device)

        pid = args.re_person
        lab = args.re_label

        person_list = [pid]
        sample_person = torch.LongTensor(person_list).to(device)
        sample_label = [lab]
        sample_label = torch.LongTensor(sample_label).to(device)
        z0 = torch.zeros(1,1).to(device).type(torch.cuda.FloatTensor)
        imggens = []
        lats = []
        with torch.no_grad():
            for ff in range(nframes):
                img_gen, lat = g_ema(sample_person, latn, class_label=sample_label, transition_factor=1, timepoints=z0+ff, input_is_latent=True, return_latents=True)
                imggens.append(img_gen)
                l = lat[0,0,:]
                lats.append(l.detach().cpu().numpy())
            img_gen = torch.stack(imggens).squeeze(1)
            img_ar = make_image(img_gen)
            outdir = outdir+"/proj/"+str(pid)+"_"+str(lab)+"/"
            Path(outdir).mkdir(parents=True, exist_ok=True)
       

        print(outdir)
        for i in range(nframes):
            PIL.Image.fromarray(img_ar[i]).save(outdir+ str(i).zfill(4)+'.png')

        
