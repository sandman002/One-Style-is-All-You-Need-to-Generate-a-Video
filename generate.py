'''
Author: Sandeep Manandhar,PhD
IBENS, ENS, Paris
'''

import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
import numpy as np
torch.manual_seed(139)

from pathlib import Path




def generate(args, generator, device, person_label, label=0, temporal_noise=None, suffix='0', dirname='/'):
    
    with torch.no_grad():
        
        g_ema.eval()
        if temporal_noise == None:
            temporal_noise = torch.randn(1, args.dyn_size, device=device)

        for i in tqdm(range(args.pics)):
            
            gen_tau = (torch.cuda.FloatTensor(torch.randint(0, 1, (1,1), device=device).type(torch.cuda.FloatTensor)))
            #gen_tau is always 0, choose the max range to higher integer to randomize that starting point
            class_labels = torch.tensor([label]).to(device)
    
            pp = person_label
            person_labels = torch.tensor([person_label]).to(device)
  
            
            savepath1 = args.savepath + '/' + dirname +'_'+str(person_label).zfill(2)+"_"+str(label).zfill(2)+'/'

            print("save location: ", savepath1)
            Path(savepath1).mkdir(parents=True, exist_ok=True)
   
            for t in range(0, args.frames):
         
                sample,lat= generator(person_labels, temporal_noise,gen_tau+t, class_label =class_labels, transition_factor =1, return_latents=True)

                utils.save_image(
                    sample,
                    savepath1+f"/frame_{str(t).zfill(5)}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                #not saving latent vector


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")
    parser.add_argument('--savepath', type=str, default='./sample/test/', help='save path for generated samples')
    parser.add_argument(
        "--size", type=int, default=128, help="output image size of the generator"
    )

    parser.add_argument(
        "--pics", type=int, default=1, help="number of images to be generated"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="./ckpt/100000.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
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

    parser.add_argument('--personlist', action='store', dest='plist',
                    type=str, nargs='*', default=['1', '2', '8'],
                    help="Examples: -i item1 item2, -i item3")

    parser.add_argument('--actlist', action='store', dest='alist',
                    type=str, nargs='*', default=['1', '2', '8'],
                    help="Examples: -i item1 item2, -i item3")


    args = parser.parse_args()
    assert len(args.plist) == len(args.alist),"person list and act list are not of equal length"
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent- args.dyn_size, args.num_person,
        args.num_classes, args.latent-args.dyn_size,  #num_classes, and c_dim
        args.dyn_size, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)

    print("loading model from:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

    try:
        ckpt_name = os.path.basename(args.ckpt)
        args.start_iter = int(os.path.splitext(ckpt_name)[0])

    except ValueError:
        pass
    
    g_ema.load_state_dict(ckpt["g_ema"])

    #################Input to generator#################
    
    person_id = [int(x) for x in args.plist]
    emo_id = [int(x) for x in args.alist]

    print(person_id, emo_id)

    count = 0
    num_temp_styles = 10 
         
    for nn in range(num_temp_styles):
        temporal_noise = torch.randn(1, args.dyn_size, device=device)
        for pp in range(len(person_id)):
            generate(args, g_ema, device, person_id[pp], emo_id[pp], temporal_noise, suffix='_'+str(count).zfill(4), dirname='t'+str(nn).zfill(2))
            count=count+1
   
