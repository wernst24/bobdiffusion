# necessary imports
import cv2 as cv
import numpy as np
from skimage import color
from skimage.filters import gaussian
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
# from scipy.ndimage import convolve
# from scipy.signal import gaussian

"""
For skimage.filters.gaussian(), truncate at (pixels away from center)/sigma
"""

def downscale_image(image, block_size):
    return downscale_local_mean(image, (block_size, block_size))

def generate_gaussian_kernel(chunk_size, sigma):
    """
    Create a 2D Gaussian kernel.
    """
    x = np.arange(-chunk_size // 2 + 1, chunk_size // 2 + 1)
    y = np.arange(-chunk_size // 2 + 1, chunk_size // 2 + 1)
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def compute_structure_tensor(image, chunk_size=5, sigma=2):
    grad_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    
    height, width = image.shape[:2]
    tensor_list = []
    gaussian_kernel = generate_gaussian_kernel(chunk_size, sigma)
    
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            chunk_grad_x = grad_x[y:y + chunk_size, x:x + chunk_size]
            chunk_grad_y = grad_y[y:y + chunk_size, x:x + chunk_size]
            
            J_xx = np.sum(chunk_grad_x ** 2 * gaussian_kernel)
            J_yy = np.sum(chunk_grad_y ** 2 * gaussian_kernel)
            J_xy = np.sum(chunk_grad_x * chunk_grad_y * gaussian_kernel)
            
            tensor_list.append(((x, y), np.array([[J_xx, J_xy], [J_xy, J_yy]])))
    return tensor_list

def calculate_orientation(tensor):
    # lowkey just like self-explanitory lock in
    _, eigvecs = np.linalg.eigh(tensor)  # Eigenvectors
    dominant_orientation = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])  # Angle of the largest eigenvector
    return np.degrees(dominant_orientation)


def sobel(image):
    return cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3), cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)

def coh_ang_calc(image, gradient_calc=sobel, sigma_inner=2, epsilon=1e-6, kernel_radius=3):
    # image: 2d grayscale image, perchance already mean downscaled a bit
    # sigma_outer: sigma for gradient detection
    # sigma_inner: sigma controlling bandwidth of angles detected
    # epsilon: prevent div0 error for coherence
    # kernel_radius: kernel size for gaussians - kernel will be 2*kernel_radius + 1 wide

    # image gradient in x and y
    I_x, I_y = gradient_calc(image)

    # structure tensor
    mu_20 = I_x ** 2
    mu_02 = I_y ** 2
    k_20_im = 2 * I_x * I_y # for later
    del I_x, I_y

    # sum bout complex representation of the 2D Structure Tensor
    k_20_re = mu_20 - mu_02
    k_11 = mu_20 + mu_02
    del mu_20, mu_02

    # this is sampling local area with w(p)
    max_std = 3.0 # cut off gaussian after 3 standard deviations
    k_20_re = gaussian(k_20_re, sigma=sigma_inner, truncate=max_std)
    k_20_im = gaussian(k_20_im, sigma=sigma_inner, truncate=max_std)
    k_11 = gaussian(k_11, sigma=sigma_inner, truncate=max_std)

    # return coherence (|k_20|/k_11), orientation (angle of k_20)
    return (k_20_re ** 2 + k_20_im ** 2) / (k_11 + epsilon) ** 2, np.arctan2(k_20_im, k_20_re)

def orient_hsv(image, coherence_image, angle_img, mode="all"):

    hsv_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    # print(np.max(angle_img), np.median(angle_img), np.min(angle_img))
    hue_img = (angle_img + np.pi) / (np.pi * 2)

    if mode == 'all':
        hsv_image[:, :, 0] = hue_img  # Hue: Orientation
        hsv_image[:, :, 1] = coherence_image # Saturation: Coherence
        hsv_image[:, :, 2] = image  # Value: Intensity
        
    elif mode == 'coherence':
        hsv_image[:, :, 0] = 0
        hsv_image[:, :, 1] = 0
        hsv_image[:, :, 2] = coherence_image # * np.mean(normalized_image[chunk_y, chunk_x])

    elif mode == 'angle':
        hsv_image[:, :, 0] = hue_img  # Hue: Orientation
        hsv_image[:, :, 1] = coherence_image
        hsv_image[:, :, 2] = 1
    else:
        assert False, "Invalid mode"

    return cv.cvtColor((hsv_image * 255).astype(np.uint8), cv.COLOR_HSV2RGB)