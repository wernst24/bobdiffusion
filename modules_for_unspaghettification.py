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


def plot_angle_histogram(weighted_hist, sigma):
    blurred = (
        gaussian_filter1d(weighted_hist, sigma=sigma, mode="wrap")
        if sigma != 0
        else weighted_hist
    )
    fig, ax = plt.subplots()
    ax.bar(
        np.linspace(0, 180, st.session_state.num_bins),
        blurred,
        width=1,
        color=matplotlib.colormaps["hsv"](np.linspace(0, 1, st.session_state.num_bins)),
    )
    ax.set_title("Coherence-Weighted Histogram of Angles")
    ax.set_ylabel("Sum of coherence (roughly counts)")
    ax.set_xlabel("Angle [deg] - 0 vertical, increasing CW")
    plt.xticks([0, 45, 90, 135, 180])
    st.pyplot(fig)


def collect_params_from_user():
    st.write("Parameters for coherence/angle calculation & display")
    # These don't need checking for NaN, because they have default values
    # st.session_state.ksize = st.number_input(value=1, min_value=1, max_value=51, step=2, label="kernel diameter")
    # st.session_state.coherence_gamma = st.number_input(
    #     value=1.0,
    #     min_value=0.5,
    #     max_value=4.0,
    #     step=0.0,
    #     label="coherence gamma",
    #     format="%.6f",
    # )



    # note: doesn't allow for ultra precise or really small sigma values - sometimes sub-pixel
    st.session_state.histogram_blur_sigma = st.number_input(
        value=0.0,
        min_value=0.0,
        max_value=100.0,
        step=0.0,
        label="histogram blur sigma",
        format="%.6f",
    )

    st.session_state.sigma_to_ydim_ratio = st.number_input(


        value=4.0 / st.session_state.raw_image_gray.shape[0],
        min_value=0.25 / st.session_state.raw_image_gray.shape[0],
        max_value=100 / st.session_state.raw_image_gray.shape[0],


        label="outerSigma to ydim ratio",
        format="%.6f",
    )

    """
    # old input for sigma value
    st.session_state.innerSigma_to_ydim_ratio = st.number_input(
        value=4.0 / st.session_state.raw_image_gray.shape[0],
        min_value=0.25 / st.session_state.raw_image_gray.shape[0],
        max_value=100 / st.session_state.raw_image_gray.shape[0],
        label="innerSigma to ydim ratio",
        format="%.6f",
    )
    """

    st.session_state.innerSigma_to_ydim_ratio = st.number_input(
        value=4.0 / st.session_state.raw_image_gray.shape[0],
        min_value=0.25 / st.session_state.raw_image_gray.shape[0],
        max_value=100.0 / st.session_state.raw_image_gray.shape[0],
        label="innerSigma to ydim ratio",
        format="%.6f",
    )

    st.session_state.epsilon = st.number_input(
        value=1.0,
        min_value=1e-3,
        max_value=100.0,
        step=-1.0,
        label="epsilon (to be divided by (ydim^2) for scale invariance)",
        format="%.6f",
    )

    st.session_state.num_bins = st.number_input(
        value=180, min_value=180, max_value=5000, step=0, label="num_bins"
    )


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
