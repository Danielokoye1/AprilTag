import numpy as np
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pupil_apriltags import Detector

# Eval
def eval_image(ref, img):
    # Calculate PSNR
    psnr = peak_signal_noise_ratio(ref, img)

    # Calculate SSIM
    ssim = structural_similarity(ref, img)

    return psnr, ssim

# Angle Error
def angle_error(est, true):
    diff_wrapped = (est - true + 180) % 360 - 180
    return abs(diff_wrapped)

# AprilTag Detection and Angle Estimation
def detect_apriltag_angle(detector, img):
    # Detect AprilTag
    tags = detector.detect(img)

    if len(tags) == 0:
        return None  # No tag detected

    # Assuming only one tag is present, take the first detected tag
    tag = tags[0]

    p0, p1, p2, p3 = tag.corners
    c = tag.center

    ang1 = ((np.rad2deg(np.arctan2(p0[1] - c[1], p0[0] - c[0])) - 135.0 + 180) % 360) - 180
    ang2 = ((np.rad2deg(np.arctan2(p1[1] - c[1], p1[0] - c[0])) - 45.0 + 180) % 360) - 180
    ang3 = ((np.rad2deg(np.arctan2(p2[1] - c[1], p2[0] - c[0])) + 45.0 + 180) % 360) - 180
    ang4 = ((np.rad2deg(np.arctan2(p3[1] - c[1], p3[0] - c[0])) + 135.0 + 180) % 360) - 180

    angs = [ang1, ang2, ang3, ang4]

    if max(angs) - min(angs) > 300:
        total = 0.0
        for a in angs:
            total += np.sign(a) * 180 - a
        avg = total / 4.0
        angle = np.sign(avg) * 180 - np.sign(avg) * avg
    else:
        angle = np.mean(angs)

    return float(angle)

# Stand-in for velocity estimate/ground truth
vel_est = 0
vel_tr = 0

# Select Folder
folder = '45_deg_roll_flat/'
angle = 45

# Reference Image as greyscale
ref = imread(folder + 'sharp_midpoint.png', as_gray=True)

# Comparison Image (Replace with Deblurred Image as needed)
comp_image = 'linear_008px.png'
img = imread(folder + comp_image, as_gray=True)

# Compare images PSNR and SSIM
psnr,ssim = eval_image(ref,img)

# Detect AprilTag and estimate rotation angle
detector = Detector(families='tag36h11')

angle_est = detect_apriltag_angle(detector, img)

print('PSNR:',psnr)
print('SSIM:',ssim)

# Print angle error
if (angle_est == None):
    print('AprilTag not detected')
else:
    print('Angle Error:', angle_error(angle_est, angle))
    if angle != 0:
        print('Angle Percent Error:', angle_error(angle_est, angle)/angle*100)

# Print vel error
print('Vel Error:', abs(vel_est - vel_tr))
if vel_tr != 0:
    print('Vel Percent Error:', abs(vel_est - vel_tr)/vel_tr*100)