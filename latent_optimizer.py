# Generate images using pretrained network pickle.
import os
import numpy as np
import PIL.Image
import argparse

import dnnlib
import torch
import loader

from training import misc
from training.misc import crop_max_rectangle as crop

from torch import optim
from tqdm import tqdm
from clip_loss import CLIPLoss
from training.networks import Generator
import clip
import math
import torchvision

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

# Generate images using pretrained network pickle.
def run(model, gpus, output_dir, images_num, truncation_psi, ratio, text, step, lr):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus                             # Set GPUs
    device = torch.device("cuda")

    print("Loading networks...")
    G = loader.load_network(model, eval = True)["Gs"].to(device)          # Load pre-trained network

    print("Generate and save images...")
    os.makedirs(output_dir, exist_ok = True)                              # Make output directory
    
    text_inputs = torch.cat([clip.tokenize(text)]).cuda()

    for i in range(images_num):
        z = torch.randn([1, *G.input_shape[1:]], device = device)         # Sample latent vector

        latent = z.detach().clone()
        latent.requires_grad = True

        clip_loss = CLIPLoss()
        optimizer = optim.Adam([latent])

        pbar = tqdm(range(step))

        for i in pbar:
            t = i / step
            lr = get_lr(t, lr)
            optimizer.param_groups[0]["lr"] = lr

            img_gen = G(latent, truncation_psi=truncation_psi)[0] # Generate an image

            c_loss = clip_loss(img_gen, text_inputs)
            loss = c_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description((f"loss: {loss.item():.4f};"))

            with torch.no_grad():
                img_gen = G(latent, truncation_psi=truncation_psi)[0]
            imgs = img_gen.cpu().numpy()
            pattern = "{}/sample_{{:06d}}.png".format(output_dir)  # Output images pattern
            img = crop(misc.to_pil(imgs[0]), ratio).save(pattern.format(i))  # Save the image

        imgs = img_gen.cpu().numpy()    # Generate an image
        pattern = "{}/sample_{{:06d}}.png".format(output_dir)             # Output images pattern
        img = crop(misc.to_pil(imgs[0]), ratio).save(pattern.format(i))   # Save the image

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description = "Generate images with the GANformer")
    parser.add_argument("--model",              help = "Filename for a snapshot to resume", type = str)
    parser.add_argument("--gpus",               help = "Comma-separated list of GPUs to be used (default: %(default)s)", default = "0", type = str)
    parser.add_argument("--output-dir",         help = "Root directory for experiments (default: %(default)s)", default = "images", metavar = "DIR")
    parser.add_argument("--images-num",         help = "Number of images to generate (default: %(default)s)", default = 32, type = int)
    parser.add_argument("--truncation-psi",     help = "Truncation Psi to be used in producing sample images (default: %(default)s)", default = 0.7, type = float)
    parser.add_argument("--ratio",              help = "Crop ratio for output images (default: %(default)s)", default = 1.0, type = float)
    parser.add_argument("--text",               help="Filename for a snapshot to resume (default: %(default)s)", default="A person with purple hair", type=str)
    parser.add_argument("--step",               help = "Number of optimization steps  (default: %(default)s)", default=40, type=float)
    parser.add_argument("--lr",                 help="Number of optimization steps  (default: %(default)s)", default=0.1, type=float)
    # Pretrained models' ratios: CLEVR (0.75), Bedrooms (188/256), Cityscapes (0.5), FFHQ (1.0)
    args, _ = parser.parse_known_args()
    run(**vars(args))

if __name__ == "__main__":
    main()
