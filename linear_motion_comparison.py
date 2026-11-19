import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.restoration import wiener
from skimage.restoration import richardson_lucy
from pupil_apriltags import Detector
from file_eval import eval_image

# Define Parameters
folder = 'Project/AprilTag/BlurredAprilTags/'
comp_image = 'diagonal_32pixel.png'
ref = imread(folder + 'diagonal_0pixel.png', as_gray=True)
blur_type = 'diagonal'                  # 'horizontal', 'vertical', or 'diagonal'
motion_pixels = 32
sigma = list(range(0, 28, 2))            # Noise stddev
K = 0.005                                # Wiener SNR
iterations = 20                          # Richardson-Lucy iterations

# Helper Functions
def deblur_image(img, motion_pixels, sigma, K):
    # adding noise to the image 
    rng = np.random.default_rng(0)
    img_n = np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.float32)

    # Define Kernel for Wiener Deconvolution
    blurred_pixels = motion_pixels + 1

    if blur_type == 'horizontal':
        kernel = np.ones((1, blurred_pixels), dtype=np.float32) / blurred_pixels
    elif blur_type == 'vertical':
        kernel = np.ones((blurred_pixels, 1), dtype=np.float32) / blurred_pixels
    elif blur_type == 'diagonal':
        kernel = np.eye(blurred_pixels, dtype=np.float32) / blurred_pixels

    # Wiener Deconvolution
    pad = motion_pixels     # pad image to remove edge effects of the deconvolution
    img_pad = np.pad(img_n /255.0, pad, mode='constant', constant_values=1.0)
    reg = np.zeros((3, 3))
    reg[1, 1] = 1
    restoredW = wiener(img_pad, kernel, balance=K, reg=reg, clip=False)
    restoredW = restoredW[pad:-pad, pad:-pad] * 255.0
    restoredW = np.nan_to_num(restoredW, nan=255.0)            # NaN -> white (background)
    restoredW = np.clip(restoredW, 0, 255).astype(np.uint8)

    psnr_b, ssim_b = eval_image(ref, img_n)
    psnr_r, ssim_r = eval_image(ref, restoredW.astype(np.float32))

    # Richardson-Lucy Deconvolution
    pad = motion_pixels     # pad image to remove edge effects of the deconvolution
    img_pad = np.pad(img_n /255.0, pad, mode='constant', constant_values=1.0)
    restoredRL = richardson_lucy(img_pad, kernel, num_iter=iterations, clip=False)
    restoredRL = restoredRL[pad:-pad, pad:-pad] * 255.0

    restoredRL = np.nan_to_num(restoredRL, nan=255.0)            # NaN -> white (background)
    restoredRL = np.clip(restoredRL, 0, 255).astype(np.uint8)

    psnr_rl, ssim_rl = eval_image(ref, restoredRL.astype(np.float32))

    return restoredW, restoredRL, img_n, psnr_b, ssim_b, psnr_r, ssim_r, psnr_rl, ssim_rl


def detect_apriltag_center(detector, image, tag_id):
    tags = detector.detect(image.astype(np.uint8))
    for tag in tags:
        if tag.tag_id == tag_id:
            return tag.center
    return None


def position_error(detected_center, true_center):
    if detected_center is None:
        return None
    dx = detected_center[0] - true_center[0]
    dy = detected_center[1] - true_center[1]
    return np.sqrt(dx**2 + dy**2)

img = imread(folder + comp_image, as_gray=True).astype(np.float32)

if img.max() <= 1:
    img = img * 255

if ref.max() <= 1:
    ref = ref * 255

# detect the AprilTag center from sharp reference image
detector = Detector(families='tag36h11')
reference_tags = detector.detect(ref.astype(np.uint8))

true_tag = reference_tags[0]
tag_id = true_tag.tag_id
true_center = true_tag.center

with open('results_' + blur_type + '_' + str(motion_pixels) + 'pixel.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Noise Sigma', 'PSNR Blurred', 'SSIM Blurred',
                     'PSNR Wiener', 'SSIM Wiener', 'PSNR RL', 'SSIM RL',
                     'Position Error Blurred (pixels)', 'Position Error Wiener (pixels)',
                     'Position Error RL (pixels)', 'Detected Blurred', 'Detected Wiener', 'Detected RL'])
    for s in sigma:
        restoredW, restoredRL, img_n, psnr_b, ssim_b, psnr_r, ssim_r, psnr_rl, ssim_rl = deblur_image(img, motion_pixels, s, K)

        center_b = detect_apriltag_center(detector, img_n, tag_id)
        center_w = detect_apriltag_center(detector, restoredW, tag_id)
        center_rl = detect_apriltag_center(detector, restoredRL, tag_id)

        err_b = position_error(center_b, true_center)
        err_w = position_error(center_w, true_center)
        err_rl = position_error(center_rl, true_center)

        writer.writerow([s, psnr_b, ssim_b, psnr_r, ssim_r, psnr_rl, ssim_rl,
                         err_b, err_w, err_rl,
                         int(center_b is not None), int(center_w is not None), int(center_rl is not None)])

restoredW, restoredRL, img_n, psnr_b, ssim_b, psnr_r, ssim_r, psnr_rl, ssim_rl = deblur_image(img, motion_pixels, 0, K)
cv2.imwrite(folder + 'restored_wiener.png', restoredW)
cv2.imwrite(folder + 'restored_rl.png', restoredRL)

restoredW, restoredRL, img_n, psnr_b, ssim_b, psnr_r, ssim_r, psnr_rl, ssim_rl = deblur_image(img, motion_pixels, 5, K)
cv2.imwrite(folder + 'restored_wiener_5n.png', restoredW)
cv2.imwrite(folder + 'restored_rl_5n.png', restoredRL)

restoredW, restoredRL, img_n, psnr_b, ssim_b, psnr_r, ssim_r, psnr_rl, ssim_rl = deblur_image(img, motion_pixels, 10, K)
cv2.imwrite(folder + 'restored_wiener_10n.png', restoredW)
cv2.imwrite(folder + 'restored_rl_10n.png', restoredRL)


# PLOTS (both the images and the graphs)
restoredW, restoredRL, img_n, psnr_b, ssim_b, psnr_w, ssim_w, psnr_rl, ssim_rl = deblur_image(img, motion_pixels, 0, K)

fig, ax = plt.subplots(1, 4, figsize=(16, 5), num=1)

ax[0].imshow(ref, cmap='gray', vmin=0, vmax=255)
ax[0].set_title('Original')

ax[1].imshow(img_n, cmap='gray', vmin=0, vmax=255)
ax[1].set_title('Blurred')

ax[2].imshow(restoredW, cmap='gray', vmin=0, vmax=255)
ax[2].set_title('Wiener Restored')

ax[3].imshow(restoredRL, cmap='gray', vmin=0, vmax=255)
ax[3].set_title('Richardson-Lucy Restored')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

# Plot Results
filename = 'results_' + blur_type + '_' + str(motion_pixels) + 'pixel.csv'
data = np.genfromtxt(filename, delimiter=',', skip_header=1)

noise = data[:, 0]

# PSNR Values
psnr_b = data[:, 1]
psnr_w = data[:, 3]
psnr_rl = data[:, 5]

# SSIM Values
ssim_b = data[:, 2]
ssim_w = data[:, 4]
ssim_rl = data[:, 6]

# Position Error Values
error_b = data[:, 7]
error_w = data[:, 8]
error_rl = data[:, 9]

# Create Figure
fig, ax = plt.subplots(1, 3, figsize=(18, 5), num=2)

fig.suptitle('Linear Motion Deblurring Results for ' + blur_type +
             ' ' + str(motion_pixels) + ' Pixel Blur, with respect to noise')

# PSNR Graph
ax[0].plot(noise, psnr_b, 'o--', label='Blurred')
ax[0].plot(noise, psnr_w, 's-', label='Wiener restored')
ax[0].plot(noise, psnr_rl, '^-', label='Richardson-Lucy restored')

ax[0].set_title('PSNR (dB)')
ax[0].set_xlabel('Noise std dev')
ax[0].set_ylabel('PSNR (dB)')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

# SSIM Graph
ax[1].plot(noise, ssim_b, 'o--', label='Blurred')
ax[1].plot(noise, ssim_w, 's-', label='Wiener restored')
ax[1].plot(noise, ssim_rl, '^-', label='Richardson-Lucy restored')

ax[1].set_title('SSIM')
ax[1].set_xlabel('Noise std dev')
ax[1].set_ylabel('SSIM')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

# position error graph
ax[2].plot(noise, error_b, 'o--', label='Blurred')
ax[2].plot(noise, error_w, 's-', label='Wiener restored')
ax[2].plot(noise, error_rl, '^-', label='Richardson-Lucy restored')

# Mark where AprilTag was not detected
ax[2].plot(noise[np.isnan(error_b)],       # use isnan to find when teh tag is not detected
           np.zeros(sum(np.isnan(error_b))),
           'x', color='tab:blue', markersize=9, label='Blurred: not detected')

ax[2].plot(noise[np.isnan(error_w)],
           np.zeros(sum(np.isnan(error_w))),
           'x', color='tab:orange', markersize=9, label='Wiener: not detected')

ax[2].plot(noise[np.isnan(error_rl)],
           np.zeros(sum(np.isnan(error_rl))),
           'x', color='tab:green', markersize=9, label='RL: not detected')

ax[2].set_title('Position Error (pixels)')
ax[2].set_xlabel('Noise std dev')
ax[2].set_ylabel('Position Error (pixels)')
ax[2].legend()
ax[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_deblurring_results.png', dpi=300)
plt.show()