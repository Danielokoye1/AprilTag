import numpy as np
import cv2
from skimage.io import imread
from scipy.signal import wiener
from pupil_apriltags import Detector

# Define Parameters
folder = '45_deg_roll_flat/'
comp_image = 'roll_012deg.png'
sweep_deg = 12
sigma = 5           # Noise stddev
K = 0.005           # Wiener SNR

# Read Image
img = imread(folder + comp_image, as_gray=True).astype(np.float32)
height, width = img.shape[:2]

# Adding noise to the image to simulate camera
rng = np.random.default_rng(0)
img_n = np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.float32) # Adding noise to the image

center = (width / 2, height / 2)
max_radius = height/2

img_pn = cv2.warpPolar(img_n, dsize=None, center=center, maxRadius=max_radius, flags=cv2.WARP_FILL_OUTLIERS)

pol_h, pol_w = img_pn.shape[:2]

# Define Kernel for Wiener Deconvolution
kernel = np.zeros(pol_h, dtype=np.float32)
blurred_rows = int(pol_h * (sweep_deg / 360.0))
kernel[:blurred_rows] = 1.0 / blurred_rows
kernel = np.roll(kernel, -(blurred_rows // 2))

# Deconvolve
H = np.fft.fft(kernel)[:, None]                 # kernel spectrum, one column
G = np.fft.fft(img_pn, axis=0)                   # image spectrum, column by column
F = np.conj(H) / (np.abs(H) ** 2 + K) * G
restored_polar = np.real(np.fft.ifft(F, axis=0)).astype(np.float32)