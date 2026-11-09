import glob
import numpy as np
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Eval
def eval_image(ref, img):
    # Calculate PSNR
    psnr = peak_signal_noise_ratio(ref, img)

    # Calculate SSIM
    ssim = structural_similarity(ref, img)

    return psnr, ssim

# Select Folder
folder = '45_deg_roll_flat/'
angle = 45

# Reference Image as greyscale
ref = imread(folder + 'sharp_midpoint.png', as_gray=True)

# Comparison Image
comp_image = 'linear_008px.png'
img = imread(folder + comp_image, as_gray=True)

# Compare images PSNR and SSIM
psnr,ssim = eval_image(ref,img)

