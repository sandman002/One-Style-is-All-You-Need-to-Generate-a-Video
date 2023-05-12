import argparse
import math
import random
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils

from tqdm import tqdm
import torchvision
from torch.autograd import Variable

from torch.utils.data import DataLoader

from videoDataset.datasets_person import VideoLabelDataset
from videoDataset import transforms

from model import Generator, DiscriminatorConPerson, Discriminator_FRAMES, SineActivation
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths





def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def cycle(iterable):
    while True:
        for x in iterable:
            yield x



def train(args, loader, generator, discriminator, discriminatorFRAMES, g_optim, d_optim, g_ema, device):

    pbar = range(args.iter)
    if not os.path.exists(args.chkpath):
        os.makedirs(args.chkpath)
    if not os.path.exists(args.samplepath):
        os.makedirs(args.samplepath)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0
    lambda_transition = 0
    d_loss_val = 0
    d2_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}


    g_module = generator
    d_module = discriminator
    d2_module = discriminatorFRAMES

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_tau = (torch.cuda.FloatTensor(torch.randint(1, 45, (4,1), device=device).type(torch.cuda.FloatTensor)))


    person_list=[0,0,1,1]  #samples to generate while training
    emo_list =  [0,0,1,1] 

    
    sample_person = torch.LongTensor(person_list).to(device)
    sample_label = torch.LongTensor(emo_list).to(device)
    sample_dyn_noise = torch.randn(4, args.dyn_size, device=device)


   
    offset = [1,2,3,4] #frame offset
    kk = 0
    transition_denominator = 1/(args.transit_end - args.transit_start+1e-7)
    for it_id in pbar:
        i = it_id + args.start_iter

        if i > args.iter:
            print("Done!")

            break
        videoData, person_labels, class_labels = next(iter((loader)))  #vid_span: original video length in the data folder
        data     = videoData[0] 
        idx      = videoData[1] 
        vid_span = videoData[2] 

        vid_span = vid_span.to(device)
        idx = idx.unsqueeze(1).to(device).type(torch.cuda.FloatTensor) #needed for valid shape to make tau arrays
        class_labels = class_labels.to(device)
        person_labels = person_labels.to(device)
        frame_m1 = data[:,:,0,:,:].to(device).type(torch.cuda.FloatTensor)
        frame_0 =  data[:,:,1*offset[kk],:,:].to(device).type(torch.cuda.FloatTensor)
        frame_p1 = data[:,:,2*offset[kk],:,:].to(device).type(torch.cuda.FloatTensor)


        requires_grad(generator, False)
        requires_grad(discriminator, True)
        requires_grad(discriminatorFRAMES, True)

        gen_tau = torch.zeros(args.batch,1).to(device).type(torch.cuda.FloatTensor)
    
        for jj in range(args.batch):
            gen_tau[jj,0] = Variable(torch.cuda.FloatTensor(torch.randint(offset[kk], vid_span[jj]-offset[kk], (1,1), device=device).type(torch.cuda.FloatTensor)))
        
        zm1 = ((gen_tau - offset[kk]))
        z0  = ((gen_tau))
        zp1 = ((gen_tau + offset[kk]))

        noise = torch.randn(args.batch, args.latent-args.dyn_size, device=device)
        dyn_noise = torch.randn(args.batch, args.dyn_size, device=device)

        fake_img_m1, _ = generator(person_labels, dyn_noise, zm1, class_label =class_labels, transition_factor = lambda_transition)
        fake_img_0, _  = generator(person_labels, dyn_noise, z0,  class_label =class_labels, transition_factor = lambda_transition)
        fake_img_p1, _ = generator(person_labels, dyn_noise, zp1, class_label =class_labels, transition_factor = lambda_transition)

        if args.augment:
            frame_m1_aug, _ = augment(frame_m1, ada_aug_p)
            frame_0_aug, _ = augment(frame_0, ada_aug_p)
            frame_p1_aug, _ = augment(frame_p1, ada_aug_p)
            fake_img_m1, _ = augment(fake_img_m1, ada_aug_p)
            fake_img_0, _ = augment(fake_img_0, ada_aug_p)
            fake_img_p1, _ = augment(fake_img_p1, ada_aug_p)

        else:
            frame_m1_aug=frame_m1
            frame_0_aug =frame_0
            frame_p1_aug=frame_p1

        fake4dis = torch.cat([fake_img_m1, fake_img_0, fake_img_p1], 0)
        fake_pred = discriminator(fake4dis, class_labels, person_labels,  lambda_transition, (gen_tau), offset[kk])
        
        read4dis = torch.cat([frame_m1_aug, frame_0_aug, frame_p1_aug], 0)
        real_pred = discriminator(read4dis, class_labels,person_labels,  lambda_transition,(idx), offset[kk])


        
        idxm1 = torch.randperm(fake_img_m1.size(0))  #changing the order of initial frames
        fake_ordered = fake_img_m1[idxm1, :]

        fake_all = torch.stack([fake_ordered, fake_img_0, fake_img_p1], 2)


        fake_all = Variable(fake_all.view(args.batch*3, 3, args.size, args.size), requires_grad=True)
        real_all = torch.stack([frame_m1_aug, frame_0_aug, frame_p1_aug], 2)

        real_all = Variable(real_all.view(args.batch*3, 3,args.size, args.size), requires_grad=True)

        fake_predall = discriminatorFRAMES(fake_all)
        real_predall = discriminatorFRAMES(real_all)

        d2_loss = d_logistic_loss(real_predall, fake_predall)

        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_predall.mean()
        loss_dict["fake_score"] = fake_predall.mean()
        loss_dict["d2"] = d2_loss


        discriminator.zero_grad()
        discriminatorFRAMES.zero_grad()
        d_loss.backward()
        d2_loss.backward()
        d_optim.step()
        d2_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred0)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            
            if args.augment:
                frame_m1_aug, _ = augment(frame_m1, ada_aug_p)
                frame_0_aug, _ = augment(frame_0, ada_aug_p)
                frame_p1_aug, _ = augment(frame_p1, ada_aug_p)

            else:
                frame_m1_aug = frame_m1
                frame_0_aug = frame_0
                frame_p1_aug = frame_p1

            frame_m1_aug.requires_grad = True
            frame_0_aug.requires_grad = True
            frame_p1_aug.requires_grad = True

            read4dis = torch.cat([frame_m1_aug, frame_0_aug, frame_p1_aug], 0)
            real_pred = discriminator(read4dis, class_labels, person_labels, lambda_transition,(idx), offset[kk])


            r1_loss = d_r1_loss(real_pred, read4dis)
            
            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(discriminatorFRAMES, False)


        gen_tau = torch.zeros(args.batch,1).to(device).type(torch.cuda.FloatTensor)
        for jj in range(args.batch):
            gen_tau[jj,0] = Variable(torch.cuda.FloatTensor(torch.randint(offset[kk], vid_span[jj]-offset[kk], (1,1), device=device).type(torch.cuda.FloatTensor)))

        zm1 = ((gen_tau - offset[kk]))
        z0  = ((gen_tau))
        zp1 = ((gen_tau + offset[kk]))

        noise = torch.randn(args.batch, args.latent-args.dyn_size, device=device)
        dyn_noise = torch.randn(args.batch, args.dyn_size, device=device)


        fake_img_m1, latm1 = generator(person_labels, dyn_noise, timepoints=zm1, class_label =class_labels, transition_factor = lambda_transition, return_latents=True)
        fake_img_0, lat0   = generator(person_labels, dyn_noise, timepoints=z0,  class_label =class_labels, transition_factor = lambda_transition, return_latents=True)
        fake_img_p1, latp1 = generator(person_labels, dyn_noise, timepoints=zp1, class_label =class_labels, transition_factor = lambda_transition, return_latents=True)


        if args.augment:
            fake_img_m1, _ = augment(fake_img_m1, ada_aug_p)
            fake_img_0, _ = augment(fake_img_0, ada_aug_p)
            fake_img_p1, _ = augment(fake_img_p1, ada_aug_p)

        fake4dis = torch.cat([fake_img_m1, fake_img_0, fake_img_p1], 0)
        fake_pred = discriminator(fake4dis, class_labels, person_labels, lambda_transition, (gen_tau), offset[kk])

       
        fake_all = torch.stack([fake_img_m1, fake_img_0, fake_img_p1], 2)
        fake_all =fake_all.view(args.batch*3, 3,args.size, args.size)

        fake_predall = discriminatorFRAMES(fake_all)


        g_loss = g_nonsaturating_loss(fake_pred) + g_nonsaturating_loss(fake_predall)
        
        loss_dict["g"] = g_loss

        generator.zero_grad()

        g_loss.backward()
        
        g_optim.step()


        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            gen_tau = torch.zeros(path_batch_size,1).to(device).type(torch.cuda.FloatTensor)
            for jj in range(path_batch_size):
                gen_tau[jj,0] = Variable(torch.cuda.FloatTensor(torch.randint(offset[kk], vid_span[jj]-offset[kk], (1,1), device=device).type(torch.cuda.FloatTensor)))
            
            path_vidspan = vid_span[:path_batch_size]
            path_class_labels = class_labels[:path_batch_size]
            path_person_labels = person_labels[:path_batch_size]
            zm1 = ((gen_tau - offset[kk]))
            z0  = ((gen_tau))
            zp1 = ((gen_tau + offset[kk]))

            noise = torch.randn(path_batch_size, args.latent - args.dyn_size, device=device)
            dyn_noise = torch.randn(path_batch_size, args.dyn_size, device=device)

            fake_img_m1, latm1 = generator(path_person_labels, dyn_noise, timepoints=zm1, class_label =path_class_labels, transition_factor = lambda_transition, return_latents=True)
            fake_img_0, lat0 =   generator(path_person_labels, dyn_noise, timepoints=z0,  class_label =path_class_labels, transition_factor = lambda_transition, return_latents=True)
            fake_img_p1, latp1 = generator(path_person_labels, dyn_noise, timepoints=zp1, class_label =path_class_labels, transition_factor = lambda_transition, return_latents=True)

            path_lossm1, mean_path_lengthm1, path_lengthsm1 = g_path_regularize(
                fake_img_m1, latm1, mean_path_length
            )

            path_loss0, mean_path_length0, path_lengths0 = g_path_regularize(
                fake_img_0, lat0, mean_path_length
            )

            path_lossp1, mean_path_lengthp1, path_lengthsp1 = g_path_regularize(
                fake_img_p1, latp1, mean_path_length
            )

            path_loss = (path_lossm1 + path_loss0 + path_lossp1)/3.0
            generator.zero_grad()

            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img_0[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()
            mean_path_length = (mean_path_length0 + mean_path_lengthm1 + mean_path_lengthp1)/3.0
            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        if i >= args.transit_start: #begin class infulence on style
            factor = (i - args.transit_start)/(args.transit_end - args.transit_start+1e-7)
            lambda_transition = np.clip(factor, a_min=0, a_max=1)


        loss_dict["path"] = path_loss
        path_lengths = (path_lossp1 + path_loss0 + path_lossm1)/3.0
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)
        d2_loss_val = loss_reduced["d2"].mean().item()
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; d2: {d2_loss_val:.4f}; g: {g_loss_val:.4f}; t: {lambda_transition:.4f} r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            

            if(i%100==0): #changing frame offsets after 100 iteration
                kk = (kk+1)%3 #offset index

            if i % 5000 == 0: #generate samples after 5000 training iterations
                with torch.no_grad():
                    g_ema.eval()
                    
                    sample_gen_tau = torch.zeros(4,1).to(device).type(torch.cuda.FloatTensor)
                    for jj in range(4):
                        sample_gen_tau[jj,0] = Variable(torch.cuda.FloatTensor(torch.randint(offset[kk], vid_span[jj]-offset[kk], (1,1), device=device).type(torch.cuda.FloatTensor)))
                    
                    path_vidspan = vid_span[:4]
                    
                    zm1 = ((sample_gen_tau - offset[kk]))
                    z0  = ((sample_gen_tau))
                    zp1 = ((sample_gen_tau + offset[kk]))

                    samplem1, _ = g_ema(sample_person, sample_dyn_noise, class_label=sample_label, transition_factor=lambda_transition, timepoints=zm1)
                    sample0, _ =  g_ema(sample_person, sample_dyn_noise, class_label=sample_label, transition_factor=lambda_transition, timepoints=z0)
                    samplep1, _ = g_ema(sample_person, sample_dyn_noise, class_label=sample_label, transition_factor=lambda_transition, timepoints=zp1)

                    utils.save_image(
                        samplem1,
                        args.samplepath+f"/{str(i).zfill(6)}_0.png",
                        nrow=int(4 ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    utils.save_image(
                        sample0,
                        args.samplepath+f"/{str(i).zfill(6)}_1.png",
                        nrow=int(4 ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    utils.save_image(
                        samplep1,
                        args.samplepath+f"/{str(i).zfill(6)}_2.png",
                        nrow=int(4 ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )


            if i % 10000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "d2": d2_module.state_dict(),

                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "d2_optim": d2_optim.state_dict(),

                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    args.chkpath+f"/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="CTSVG trainer")
    parser.add_argument('--samplepath', type=str, default='./sample/', help='model architectures (stylegan2 | swagan)')
    parser.add_argument('--chkpath', type=str, default='./checkpoint/', help='model architectures (stylegan2 | swagan)')
    parser.add_argument('--transit_start', type=int, default=2000, help='starting iteration of conditional transition')
    parser.add_argument('--transit_end', type=int, default=4000, help='ending iteration of conditional transition')
    parser.add_argument(
        "--iter", type=int, default=150000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=4, help="batch sizes for each gpus"
    )

    parser.add_argument(
        "--size", type=int, default=128, help="image sizes for the model"
    )

    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )

    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )

    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )


    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )

     
    parser.add_argument(
        "--latent",
        type=int,
        default=320,
        help="total dim of latent vector",
    )

    parser.add_argument(
        "--dyn_size",
        type=int,
        default=256,
        help="num of t2v components",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=8,
        help="number of classes for conditional generation",
    )
    parser.add_argument(
        "--num_person",
        type=int,
        default=30,
        help="number of person for conditional generation",
    )

    parser.add_argument(
        "--traindata_csv",
        type=str,
        default="./datasets/mead_train.csv",
        help="path to the checkpoints to resume training",
    )


    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    args.distributed = n_gpu > 1
    print(n_gpu,'####################')

    args.n_mlp = 8

    args.start_iter = 0
    
   
    print('person_num, label_num: ', args.num_person, args.num_classes)
    generator = Generator(
        args.size, args.latent- args.dyn_size, args.num_person,
        args.num_classes, args.latent-args.dyn_size,  #num_classes, and c_dim
        args.dyn_size, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    discriminator = DiscriminatorConPerson(
        args.size, num_classes=args.num_classes,num_persons=args.num_person, c_dim = 128, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminatorFRAMES = Discriminator_FRAMES(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent- args.dyn_size, args.num_person,
        args.num_classes, args.latent-args.dyn_size, 
        args.dyn_size, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)



    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    d2_optim = optim.Adam(
        discriminatorFRAMES.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"], strict=False)

        discriminator.load_state_dict(ckpt["d"])
        discriminatorFRAMES.load_state_dict(ckpt["d2"], strict=False)
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])
        d2_optim.load_state_dict(ckpt["d2_optim"])

    n_cpu = 8
    
    
    trainset = VideoLabelDataset(
    args.traindata_csv,
    transform = torchvision.transforms.Compose([
                           transforms.VideoFolderPathToTensor(max_len = 9, padding_mode = 'last'),
                           transforms.VideoResize([args.size, args.size] ), #videos are normalized here so regardless of resize or not, this has to be called
                        ]))
                             # See more options at transforms.py
    
    print(len(trainset), 'size of trainset')
    train_loader = DataLoader(trainset,
                      batch_size=args.batch,
                      num_workers=n_cpu, 
                      shuffle=True,
                      drop_last = True, 
                      pin_memory=True
                      )

    dataset_size = len(train_loader.dataset)

    print('# training videos = %d' % dataset_size)


    from datetime import datetime
    import json
    now = datetime.now()
    argsavename = now.strftime("%d-%m-%Y-%H-%M-%S") + '_args.txt'
    with open(argsavename, 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)

    train(args, train_loader, generator, discriminator, discriminatorFRAMES, g_optim, d_optim, g_ema, device)



