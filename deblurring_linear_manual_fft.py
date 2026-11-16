import numpy as np
import csv
from skimage.io import imread
#from scipy.signal import wiener
from skimage.restoration import richardson_lucy, wiener
from pupil_apriltags import Detector
from file_eval import eval_image, detect_apriltag_center

# Define Parameters
folder = 'BlurredAprilTags/'
comp_image = 'horizontal_32pixel.png'

ref = imread(folder + 'sharp_tag.png', as_gray=True) #loads sharp image

#check to see if the ref range is [0 1] and if it is scale it to [0 255]
if ref.max() <= 1:
    ref = ref * 255 


ref = ref.astype(np.uint8) #store as 8bit img

blur_type = 'horizontal'       # can be 'horizontal', 'vertical', or 'diagonal'
motion_pixels = 32
#sigma = 5          # Noise stddev -> commenting out to create varying noise graphs

sigma_values = list(range(0, 28, 2)) #noise stddev

K = 0.005          # Wiener SNR
iterations = 20    # Richardson-Lucy iterations


results_file = 'results_horizontal_32pixel_updated.csv'

# Read Image
img = imread(folder + comp_image, as_gray=True).astype(np.float32)
height, width = img.shape[:2]

# check to see if the img range is [0 1] and if it is scale it to [0 255]
if img.max() <= 1:
    img = img * 255

    #calculating img true center
true_center = np.array([
    (img.shape[1] - 1) / 2,
    (img.shape[0] - 1) / 2,
])

    
#Adding noise to the image to simulate camera
#// ******COMMENTING OUT ONE TIME CALCULATION*******

# rng = np.random.default_rng(0)
# 
# img_n = np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.float32) # Adding noise to the image
# 
# # Define Kernel for Wiener Deconvolution
# blurred_pixels = motion_pixels + 1
# 
# # define the kernel based on the blur type
# if blur_type == 'horizontal':
#     kernel = np.ones((1, blurred_pixels), dtype =np.float32) / blurred_pixels
# elif blur_type == 'vertical':
#     kernel = np.ones((blurred_pixels, 1), dtype =np.float32) / blurred_pixels
# elif blur_type == 'diagonal':
#     kernel = np.eye(blurred_pixels, dtype =np.float32) / blurred_pixels
# 
# # Wiener Deconvolution
# H = np.fft.fft2(kernel, s=img_n.shape)      # Zero padding the kernel so its teh same size of the img
# G = np.fft.fft2(img_n)                      # image spectrum, column by column
# F = np.conj(H) / (np.abs(H)**2 + K) * G
# restoredWiener = np.real(np.fft.ifft2(F)).astype(np.float32); 
# shift = (kernel.shape[0] // 2, kernel.shape[1] // 2)              # Shift kernel to be centered in img
# restoredWiener = np.roll(restoredWiener, shift, axis=(0,1))       # Helps correct pos of kernel
# restoredWiener = np.clip(restoredWiener, 0, 255)                  # Some values may end outside of range so clip values
# 
# # Richardson-Lucy Deconvolution
# restoredRL = richardson_lucy(img_n, kernel, num_iter = iterations, clip=False)
# restoredRL = np.clip(restoredRL, 0, 255)


# Helper function for testing Wiener deblurring at one noise level.
def wiener_deblur_image(img_n, blur_type, motion_pixels, sigma,K,):


    #repeatable random-noise pattern.
    rng = np.random.default_rng(0)

    # add Gaussian noise using the current sigma value.
    img_n = np.clip(
        img_n + rng.normal(0, sigma, img_n.shape),
        0,
        255,
    ).astype(np.float32)

    #converting  motion distance into a kernel length.
    blurred_pixels = motion_pixels + 1

    # Horizontal motion kernel.
    if blur_type == 'horizontal':
        kernel = (
            np.ones(
             (1, blurred_pixels),
              dtype=np.float32,
            )
            / blurred_pixels
        )

    # Vertical motion kernel.
    elif blur_type == 'vertical':
        kernel = (
          np.ones(
             (blurred_pixels, 1),
              dtype=np.float32,
            )
            / blurred_pixels
        )

    # Diagonal motion kernel.
    elif blur_type == 'diagonal':
        kernel = (
            np.eye(
                blurred_pixels,
                dtype=np.float32,
            )
            / blurred_pixels
        )


    # linear-motion kernel into frequency space.
    H = np.fft.fft2(
        kernel,
        s=img_n.shape,
    )

    #  noisy blurred image into frequency space.
    G = np.fft.fft2(img_n)

    # Apply Wiener deconvolution.
    # K limits the amount of noise amplification.
    F = (
        np.conj(H)
        / (np.abs(H) ** 2 + K)
        * G
    )

    # Transform the result back into image space.
    restored = np.real(
        np.fft.ifft2(F)
    ).astype(np.float32)

    # Determine the spatial shift introduced by the kernel.
    shift = (
        kernel.shape[0] // 2,
        kernel.shape[1] // 2,
    )

    # Realign the restored image.
    restored = np.roll(
        restored,
        shift,
        axis=(0, 1),
    )

    # Keep valid image intensity values.
    restored = np.clip(
        restored,
        0,
        255,
    )

    # Compare the noisy blurred image with the sharp reference.
    psnr_b, ssim_b = eval_image(
        ref,
        img_n,
    )

    # Compare the restored image with the sharp reference.
    psnr_r, ssim_r = eval_image(
        ref,
        restored,
    )

    
    return (
        restored,
        img_n,
        psnr_b,
        ssim_b,
        psnr_r,
        ssim_r,
    )
    #init detector
detector = Detector(families='tag36h11')

# Open the results CSV.

with open(results_file, 'w', newline='') as f:

    # Create the CSV writer.
    writer = csv.writer(f)
  
    writer.writerow([
        'Noise Sigma',
        'PSNR Blurred',
        'SSIM Blurred',
        'PSNR Restored',
        'SSIM Restored',
        'Position Error Blurred',
        'Position Error Restored',
    ])


        # iterate through all noise values
    for s in sigma_values:

        # Run the linear Wiener helper at the current noise level.
        (
            restored,
            img_n,
            psnr_b,
            ssim_b,
            psnr_r,
            ssim_r,
        ) = wiener_deblur_image(
            img,
            blur_type,
            motion_pixels,
            s,
            K,
        )

        # Detect the AprilTag center before deblurring.
        center_blurred = detect_apriltag_center(
            detector,
            img_n.astype(np.uint8),
        )

        # No position error can be calculated if detection fails.
        if center_blurred is None:
            position_error_blurred = None

        # Calculate distance from the known image center.
        else:
            position_error_blurred = np.linalg.norm(
                center_blurred - true_center
            )

         # detect the AprilTag center after deblurring.
        center_restored = detect_apriltag_center(
            detector,
            restored.astype(np.uint8),
        )
        #calculates nothing if not detected

        if center_restored is None:
            position_error_restored = None

        # Calculate restored distance from the known center.
        else:
            position_error_restored = np.linalg.norm(
                center_restored - true_center
            )

          # Save one CSV row for this noise level.
        writer.writerow([
            s,
            psnr_b,
            ssim_b,
            psnr_r,
            ssim_r,
            position_error_blurred,
            position_error_restored,
        ])