import numpy as np
import cv2
from skimage.io import imread
from scipy.signal import wiener
from pupil_apriltags import Detector

# Define Image
folder = '45_deg_roll_flat/'
comp_image = 'linear_008px.png'

# Read Image
img = imread(folder + comp_image, as_gray=True).astype(np.float32)
height, width = img.shape[:2]

center = (width / 2, height / 2)
max_radius = height/2

polar_img = cv2.warpPolar(img, dsize=None, center=center, maxRadius=max_radius, flags=cv2.WARP_FILL_OUTLIERS)

cv2.imshow('Polar Image', polar_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

pol_h, pol_w = polar_img.shape[:2]

