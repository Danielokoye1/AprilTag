import numpy as np
import cv2
from skimage.io import imread
from scipy.signal import wiener
from skimage.restoration import richardson_lucy
from pupil_apriltags import Detector

# Define Parameters
folder = ''
comp_image = 'horizontal_32pixel.png'
blur_type = 'horizontal'       # can be 'horizontal', 'vertical', or 'diagonal'
motion_pixels = 32
sigma = 5          # Noise stddev
K = 0.005          # Wiener SNR
iterations = 20    # Richardson-Lucy iterations

# Read Image
img = imread(folder + comp_image, as_gray=True).astype(np.float32)
height, width = img.shape[:2]

# check to see if the img range is [0 1] and if it is scale it to [0 255]
if img.max() <= 1:
    img = img * 255

# Adding noise to the image to simulate camera
rng = np.random.default_rng(0)
img_n = np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.float32) # Adding noise to the image

# Define Kernel for Wiener Deconvolution
blurred_pixels = motion_pixels + 1

# define the kernel based on the blur type
if blur_type == 'horizontal':
    kernel = np.ones((1, blurred_pixels), dtype =np.float32) / blurred_pixels
elif blur_type == 'vertical':
    kernel = np.ones((blurred_pixels, 1), dtype =np.float32) / blurred_pixels
elif blur_type == 'diagonal':
    kernel = np.eye(blurred_pixels, dtype =np.float32) / blurred_pixels

# Wiener Deconvolution
H = np.fft.fft2(kernel, s=img_n.shape)      # Zero padding the kernel so its teh same size of the img
G = np.fft.fft2(img_n)                      # image spectrum, column by column
F = np.conj(H) / (np.abs(H)**2 + K) * G
restoredWiener = np.real(np.fft.ifft2(F)).astype(np.float32); 
shift = (kernel.shape[0] // 2, kernel.shape[1] // 2)              # Shift kernel to be centered in img
restoredWiener = np.roll(restoredWiener, shift, axis=(0,1))       # Helps correct pos of kernel
restoredWiener = np.clip(restoredWiener, 0, 255)                  # Some values may end outside of range so clip values

# Richardson-Lucy Deconvolution
restoredRL = richardson_lucy(img_n, kernel, num_iter = iterations, clip=False)
restoredRL = np.clip(restoredRL, 0, 255)