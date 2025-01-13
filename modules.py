# necessary imports
import cv2 as cv
import numpy as np
from skimage import color
from skimage.filters import gaussian
from skimage.transform import rescale
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import streamlit as st
# from io import BytesIO

def sobel(image):
    return cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5), cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)

@st.cache_data
def structure_tensor_calc(image):
    I_x, I_y = sobel(image)

    # structure tensor
    mu_20 = I_x ** 2
    mu_02 = I_y ** 2

    # k_20_real, k_20_im, k_11
    return mu_20 - mu_02, 2 * I_x * I_y, mu_20 + mu_02

def kval_gaussian(k_20_re, k_20_im, k_11, sigma):
    max_std = 3.0 # cut off gaussian after 3 standard deviations
    return gaussian(k_20_re, sigma=sigma, truncate=max_std), gaussian(k_20_im, sigma=sigma, truncate=max_std), gaussian(k_11, sigma=sigma, truncate=max_std)

@st.cache_data
def coh_ang_calc(image, sigma_inner=2, epsilon=1e-3, kernel_radius=3):
    # image: 2d grayscale image, perchance already mean downscaled a bit
    # sigma_outer: sigma for gradient detection
    # sigma_inner: sigma controlling bandwidth of angles detected
    # epsilon: prevent div0 error for coherence
    # kernel_radius: kernel size for gaussians - kernel will be 2*kernel_radius + 1 wide

    k_20_re, k_20_im, k_11 = structure_tensor_calc(image)

    # this is sampling local area with w(p)
    k_20_re, k_20_im, k_11 = kval_gaussian(k_20_re, k_20_im, k_11, sigma_inner)

    # return coherence (|k_20|/k_11), orientation (angle of k_20)
    return (k_20_re ** 2 + k_20_im ** 2) / (k_11 + epsilon) ** 2, np.arctan2(k_20_im, k_20_re)



# get rbg of image, coherence, and angle
@st.cache_data
def orient_hsv(image, coherence_image, angle_img, mode="all", angle_phase=0, invert=False):
    angle_img  = (angle_img - angle_phase*np.pi/90.0) % (np.pi * 2)

    hsv_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    # intial hue calculation - [0, 1)
    hue_img = (angle_img) / (np.pi * 3) # i dont know why this uses 3pi but it looks good

    if mode == 'all':
        hsv_image[:, :, 0] = hue_img  # Hue: Orientation
        hsv_image[:, :, 1] = coherence_image # Saturation: Coherence
        hsv_image[:, :, 2] = image if not invert else 1 - image  # Value: Intensity
        
    elif mode == 'coherence':
        hsv_image[:, :, 0] = 0
        hsv_image[:, :, 1] = 0
        hsv_image[:, :, 2] = coherence_image # * np.mean(normalized_image[chunk_y, chunk_x])

    elif mode == 'angle':
        hsv_image[:, :, 0] = hue_img  # Hue: Orientation
        hsv_image[:, :, 1] = coherence_image
        hsv_image[:, :, 2] = 1

    elif mode == 'angle_bw':
        hsv_image[:, :, 0] = 0
        hsv_image[:, :, 1] = 0
        hsv_image[:, :, 2] = (angle_img) / (2 * np.pi)
    else:
        assert False, "Invalid mode"

    return cv.cvtColor((hsv_image * 255).astype(np.uint8), cv.COLOR_HSV2RGB)
