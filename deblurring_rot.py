import numpy as np
import cv2
from skimage.io import imread
from scipy.signal import wiener
from pupil_apriltags import Detector
from file_eval import eval_image, angle_error, detect_apriltag_angle

# Define Parameters
folder = '45_deg_roll_flat/'
comp_image = 'roll_024deg.png'
sweep_deg = 24
sigma = 3           # Noise stddev
K = 0.005           # Wiener SNR
angle = 45          # Ground truth angle for comparison

# Read Image
img = imread(folder + comp_image, as_gray=True).astype(np.float32)
height, width = img.shape[:2]

# Adding noise to the image to simulate camera
rng = np.random.default_rng(0)
img_n = np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.float32) # Adding noise to the image

# Convert to Polar Coordinates
center = (width / 2, height / 2)
max_radius = height/2
flgs = cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR

img_pn = cv2.warpPolar(img_n, dsize=None, center=center, maxRadius=max_radius, flags=flgs)

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

restored = cv2.warpPolar(restored_polar, (width, height), center, max_radius, flags = flgs + cv2.WARP_INVERSE_MAP)
restored = np.clip(restored, 0, 255).astype(np.uint8)

# Detect AprilTag and estimate rotation angle
detector = Detector(families='tag36h11')

angle_est = detect_apriltag_angle(detector, img)

# print('PSNR:',psnr)
# print('SSIM:',ssim)

# Print angle error
if (angle_est == None):
    print('AprilTag not detected')
else:
    print('Angle Error:', angle_error(angle_est, angle))
    if angle != 0:
        print('Angle Percent Error:', angle_error(angle_est, angle)/angle*100)