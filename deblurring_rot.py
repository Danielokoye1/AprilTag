import numpy as np
import cv2
import csv

from skimage.io import imread
from skimage.restoration import wiener
from skimage.restoration import richardson_lucy
from pupil_apriltags import Detector
from file_eval import eval_image, angle_error, detect_apriltag_angle

# Define Parameters
folder = '45_deg_roll_flat/'
comp_image = 'roll_024deg.png'
ref = imread(folder + 'sharp_midpoint.png', as_gray=True)
sweep_deg = 24
sigma = list(range(0, 28, 2))               # Noise stddev
K = 0.005                                   # Wiener SNR
angle = 45                                  # Ground truth angle for comparison
iterations = 20                             # Richardson-Lucy iterations

# Helper Functions
def deblur_image(img_n, sweep_deg, sigma, K):
    # Convert to Polar Coordinates
    height, width = img_n.shape[:2]
    center = (width / 2, height / 2)
    max_radius = height / 2
    flgs = cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR

    # Adding noise to the image to simulate camera
    rng = np.random.default_rng(0)
    img_n = np.clip(img_n + rng.normal(0, sigma, img_n.shape), 0, 255).astype(np.float32) # Adding noise to the image

    img_pn = cv2.warpPolar(img_n, dsize=None, center=center, maxRadius=max_radius, flags=flgs)

    img_pn = cv2.warpPolar(img_n, dsize=None, center=center, maxRadius=max_radius, flags=flgs)

    pol_h, pol_w = img_pn.shape[:2]  # Shape of polar coords image

    # Define Kernel for Wiener Deconvolution
    blurred_rows = int(pol_h * (sweep_deg / 360.0))

    psf = np.ones((blurred_rows, 1)) / blurred_rows      # point spread func
    reg = np.zeros((3, 3)); reg[1, 1] = 1                # regularizer
    restored_polar = wiener(img_pn / 255.0, psf, balance=K, reg=reg, clip=False) * 255.0

    restored = cv2.warpPolar(restored_polar, (width, height), center, max_radius, flags=flgs + cv2.WARP_INVERSE_MAP)
    restored = np.nan_to_num(restored, nan=255.0)          # NaN -> white (background)
    restored = np.clip(restored, 0, 255).astype(np.uint8)

    # Cropping the image so that the dark circle outside is eliminated
    c0, r0 = int(center[0]), int(center[1])
    half = int(max_radius / np.sqrt(2))            # largest square that fits inside the circle
    crop = (slice(r0 - half, r0 + half), slice(c0 - half, c0 + half))

    
    restored = np.nan_to_num(restored, nan=255.0)          # NaN -> white (background)
    restored = np.clip(restored, 0, 255).astype(np.uint8)

    psnr_b, ssim_b = eval_image(ref[crop], img_n[crop])
    psnr_r, ssim_r = eval_image(ref[crop], restored.astype(np.float32)[crop])

    # Richardson-Lucy deconvolution
    kernel = np.ones((blurred_rows, 1), dtype =np.float32) / blurred_rows
    restoredRL = richardson_lucy(img_pn / 255.0, kernel, num_iter = iterations, clip=False)
    restoredRL = np.clip(restoredRL, 0, 255)
    restoredRL = np.nan_to_num(restoredRL, nan=255.0)
    restoredRL = np.clip(restoredRL, 0, 255).astype(np.uint8)

    psnr_rl, ssim_rl = eval_image(ref[crop], restoredRL.astype(np.float32)[crop])

    return restored, restoredRL, img_n, psnr_b, ssim_b, psnr_r, ssim_r, psnr_rl, ssim_rl

img = imread(folder + comp_image, as_gray=True).astype(np.float32)

# Detect AprilTag and estimate rotation angle
detector = Detector(families='tag36h11')

if img.max() <= 1:
    img = img * 255

with open('results_45deg_24roll.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Noise Sigma', 'PSNR Blurred', 'SSIM Blurred', 'PSNR Restored', 'SSIM Restored', 'Angle Error Blurred', 'Angle Error Restored'])
    for s in sigma:
        restored, restoredRL, img_n, psnr_b, ssim_b, psnr_r, ssim_r, psnr_rl, ssim_rl = deblur_image(img, sweep_deg, s, K)

        ang_est = detect_apriltag_angle(detector, restored.astype(np.uint8))
        ang_err = angle_error(ang_est, angle) if ang_est is not None else None

        ang_est_r = detect_apriltag_angle(detector, img_n.astype(np.uint8))
        ang_err_r = angle_error(ang_est_r, angle) if ang_est_r is not None else None
        
        writer.writerow([s, psnr_b, ssim_b, psnr_r, ssim_r, ang_err_r, ang_err])

restored, restoredRL, img_n, psnr_b, ssim_b, psnr_r, ssim_r, psnr_rl, ssim_rl = deblur_image(img, sweep_deg, 0, K)
cv2.imwrite(folder + 'restored_wiener.png', restored)
cv2.imwrite(folder + 'restored_rl.png', restoredRL)

restored, restoredRL, img_n, psnr_b, ssim_b, psnr_r, ssim_r, psnr_rl, ssim_rl = deblur_image(img, sweep_deg, 5, K)
cv2.imwrite(folder + 'restored_wiener_5n.png', restored)
cv2.imwrite(folder + 'restored_rl_5n.png', restoredRL)

restored, restoredRL, img_n, psnr_b, ssim_b, psnr_r, ssim_r, psnr_rl, ssim_rl = deblur_image(img, sweep_deg, 10, K)
cv2.imwrite(folder + 'restored_wiener_10n.png', restored)
cv2.imwrite(folder + 'restored_rl_10n.png', restoredRL)