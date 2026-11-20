import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.restoration import wiener
from skimage.restoration import richardson_lucy
from pupil_apriltags import Detector

# Define Parameters
folder = 'BlurredAprilTags/'
comp_image = 'Horiz2.png'
blur_type = 'horizontal'                  # 'horizontal', 'vertical', or 'diagonal'
motion_pixels = 20
K = 0.1                                # Wiener SNR
iterations = 2                          # Richardson-Lucy iterations

# Helper Functions
def deblur_image(img, motion_pixels, K):
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
    img_pad = np.pad(img /255.0, pad, mode='reflect')
    reg = np.zeros((3, 3))
    reg[1, 1] = 1
    restoredW = wiener(img_pad, kernel, balance=K, reg=reg, clip=False)
    restoredW = restoredW[pad:-pad, pad:-pad] * 255.0
    restoredW = np.nan_to_num(restoredW, nan=255.0)            # NaN -> white (background)
    restoredW = np.clip(restoredW, 0, 255).astype(np.uint8)

    # Richardson-Lucy Deconvolution
    #pad = motion_pixels     # pad image to remove edge effects of the deconvolution
    #img_pad = np.pad(img /255.0, pad, mode='reflect')
    restoredRL = richardson_lucy(img_pad, kernel, num_iter=iterations, clip=False)
    restoredRL = restoredRL[pad:-pad, pad:-pad] * 255.0
    restoredRL = np.nan_to_num(restoredRL, nan=255.0)            # NaN -> white (background)
    restoredRL = np.clip(restoredRL, 0, 255).astype(np.uint8)


    return restoredW, restoredRL

# read the img
img = imread(folder + comp_image, as_gray=True).astype(np.float32)

if img.max() <= 1:
    img = img * 255

restoredW, restoredRL = deblur_image(img, motion_pixels, K)

# detect the AprilTag center
detector = Detector(families='tag36h11')
tag_original = detector.detect(img.astype(np.uint8))
tag_W = detector.detect(restoredW)
tag_RL = detector.detect(restoredRL)

cv2.imwrite(folder + 'restored_wiener.png', restoredW)
cv2.imwrite(folder + 'restored_rl.png', restoredRL)


fig, ax = plt.subplots(1, 3, figsize=(15, 5), num=1)
ax[0].imshow(img, cmap='gray', vmin=0, vmax=255)
ax[0].set_title('Blurred Image')
ax[1].imshow(restoredW, cmap='gray', vmin=0, vmax=255)
ax[1].set_title('Wiener Restored')
ax[2].imshow(restoredRL, cmap='gray', vmin=0, vmax=255)
ax[2].set_title('Richardson-Lucy Restored')
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()