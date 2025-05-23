# necessary imports
from cv2 import cvtColor, COLOR_HSV2RGB
import numpy as np
from skimage.filters import gaussian

# import streamlit as st
from scipy.ndimage import gaussian_filter


def ddx_gaussian_convolve(
    image: np.ndarray, sigma=-1, sigma_to_ydim_ratio=-1, truncate=3
):
    if sigma == -1:
        if sigma_to_ydim_ratio == -1:
            sigma = 1
        else:
            sigma = image.shape[0] * sigma_to_ydim_ratio
    return gaussian_filter(
        image, sigma, order=(0, 1), truncate=truncate
    ), gaussian_filter(image, sigma, order=(1, 0), truncate=truncate)  # ix, iy


def structure_tensor_calc(image, sigma_to_ydim_ratio):
    I_x, I_y = ddx_gaussian_convolve(
        image, sigma_to_ydim_ratio=sigma_to_ydim_ratio, truncate=2
    )

    # structure tensor
    mu_20 = I_x * I_x
    mu_02 = I_y * I_y

    # k_20_real, k_20_im, k_11
    return mu_20 - mu_02, 2 * I_x * I_y, mu_20 + mu_02


def kval_gaussian(k_20_re, k_20_im, k_11, sigma):
    max_std = 3.0  # cut off gaussian after 3 standard deviations
    return (
        gaussian(k_20_re, sigma=sigma, truncate=max_std),
        gaussian(k_20_im, sigma=sigma, truncate=max_std),
        gaussian(k_11, sigma=sigma, truncate=max_std),
    )


def coh_ang_calc(image, sigma_to_ydim_ratio, innerSigma_to_ydim_ratio, epsilon=1e-3):
    # image: 2d grayscale image, perchance already mean downscaled a bit
    # sigma_outer: sigma for gradient detection
    # sigma_inner: sigma controlling bandwidth of angles detected
    # epsilon: prevent div0 error for coherence
    # kernel_radius: kernel size for gaussians - kernel will be 2*kernel_radius + 1 wide

    k_20_re, k_20_im, k_11 = structure_tensor_calc(image, sigma_to_ydim_ratio)

    # this is sampling local area with w(p)
    k_20_re, k_20_im, k_11 = kval_gaussian(
        k_20_re, k_20_im, k_11, innerSigma_to_ydim_ratio * image.shape[0]
    )

    # return coherence (|k_20|/k_11), orientation (angle of k_20)
    return (k_20_re * k_20_re + k_20_im * k_20_im) / (
        (k_11 + epsilon) * (k_11 + epsilon)
    ), np.arctan2(k_20_im, k_20_re)


def orient_hsv(
    image, coherence_image, angle_img, mode="all", angle_phase=0, invert=False
):
    angle_img = (angle_img - angle_phase * np.pi / 90.0) % (np.pi * 2)

    hsv_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    # intial hue calculation - [0, 1)
    hue_img = (angle_img) / (
        np.pi * 3
    )  # i dont know why this uses 3pi but it looks good

    if mode == "all":
        hsv_image[:, :, 0] = hue_img  # Hue: Orientation
        hsv_image[:, :, 1] = coherence_image  # Saturation: Coherence
        hsv_image[:, :, 2] = image if not invert else 1 - image  # Value: Intensity

    elif mode == "coherence":
        hsv_image[:, :, 0] = 0
        hsv_image[:, :, 1] = 0
        hsv_image[:, :, 2] = (
            coherence_image  # * np.mean(normalized_image[chunk_y, chunk_x])
        )

    elif mode == "angle":
        hsv_image[:, :, 0] = hue_img  # Hue: Orientation
        hsv_image[:, :, 1] = coherence_image
        hsv_image[:, :, 2] = 1

    elif mode == "angle_bw":
        hsv_image[:, :, 0] = 0
        hsv_image[:, :, 1] = 0
        hsv_image[:, :, 2] = (angle_img) / (2 * np.pi)
    else:
        assert False, "Invalid mode"

    return cvtColor((hsv_image * 255).astype(np.uint8), COLOR_HSV2RGB)


def weightedHistogram(coh, ang, num_bins):
    bin_width = np.pi * 2 / num_bins
    # print(bin_width)
    ang = np.floor(
        ((ang + 2 * np.pi) % (np.pi * 2)) / bin_width
    )  # should be from 0 to num_bins
    # print(ang, "ang min ang max \n\n\n", ang.min(), ang.max(), len(np.unique(ang)))
    cohang = np.stack((coh, ang), axis=-1)
    x, y, _ = cohang.shape
    flat = cohang.reshape((x * y, 2))
    flat = flat[np.argsort(flat[:, 1])]
    hist = np.zeros((num_bins))
    np.add.at(hist, flat[:, 1].astype(np.int16), flat[:, 0])
    return hist