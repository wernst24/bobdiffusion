from cv2 import imdecode
import numpy as np
import streamlit as st
from skimage.transform import resize
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter1d
import matplotlib
import matplotlib.pyplot as plt

def read_image_to_opencv(uploaded_image, rescale_factor):
    image_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    cv_image = imdecode(image_bytes, 1)
    cv_image_gray = rgb2gray(cv_image)
    # return transform.rescale(cv_image_gray, rescale_factor, anti_aliasing=True)
    return resize(cv_image_gray, (200, 200))

def plot_angle_histogram(weighted_hist, sigma):
    blurred = (
        gaussian_filter1d(weighted_hist, sigma=sigma, mode="wrap")
        if sigma != 0
        else weighted_hist
    )
    fig, ax = plt.subplots()
    ax.bar(
        np.linspace(0, 180, 200),
        blurred,
        width=1,
        color=matplotlib.colormaps["hsv"](np.linspace(0, 1, 200))
    )
    ax.set_title("Coherence-Weighted Histogram of Angles")
    ax.set_ylabel("Sum of coherence (roughly counts)")
    ax.set_xlabel("Angle [deg] - 0 vertical, increasing CW")
    plt.xticks([0, 45, 90, 135, 180])
    st.pyplot(fig)