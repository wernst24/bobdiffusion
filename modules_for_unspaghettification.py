import streamlit as st
import cv2 as cv
import numpy as np
from skimage import color, transform
from scipy.ndimage import gaussian_filter1d
import matplotlib
import matplotlib.pyplot as plt
from modules import orient_hsv, downscale_coh_ang


def read_image_to_opencv(uploaded_image, rescale_factor):
    image_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    cv_image = cv.imdecode(image_bytes, 1)
    cv_image_gray = color.rgb2gray(cv_image)
    return transform.rescale(cv_image_gray, rescale_factor, anti_aliasing=True)

def display_selected_falsecolor(
    raw_image, coherence, angle, mode="Intensity, Coherence, and Angle", blockreduce=1
):
    fancy_to_mode = {
        "Intensity, Coherence, and Angle": "all",
        "Coherence and Angle only": "angle",
        "Coherence only": "coherence",
        "Angle only": "angle_bw",
    }
    if blockreduce == 1:
        st.image(
            orient_hsv(raw_image, coherence, angle, mode=fancy_to_mode[mode]),
            use_container_width=True,
        )
    else:
        (h, w) = raw_image.shape
        raw_image_small = (
            raw_image[: h - h % blockreduce, : w - w % blockreduce]
            .reshape(h // blockreduce, blockreduce, w // blockreduce, blockreduce)
            .mean(axis=(1, 3))
        )
        coherence_small, angle_small = downscale_coh_ang(coherence, angle, blockreduce)
        display_selected_falsecolor(
            raw_image_small, coherence_small, angle_small, mode=mode, blockreduce=1
        )
