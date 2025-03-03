from modules import coh_ang_calc, weightedHistogram
from modules_for_unspaghettification import (
    collect_params_from_user,
    read_image_to_opencv,
    display_selected_falsecolor,
    plot_angle_histogram,
)
import streamlit as st
import numpy as np
import matplotlib

matplotlib.use("agg")  # Use non-GUI backend
import sys
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.dont_write_bytecode = True
sys.tracebacklimit = 0


st.set_page_config(
    page_title="BobHistogram",
    page_icon="🔬",
    layout="wide",
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("# weighted histogram")

    with st.form("form1", enter_to_submit=False, clear_on_submit=False):
        # upload and get rescale factor
        msg = "Upload a 2D image to be analyzed. Downsizing is reccomended for larger images"
        uploaded_image = st.file_uploader(
            msg, type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=False
        )
        rescale_factor = st.number_input(
            "Lazy scale space (downscale)",
            min_value=0.01,
            max_value=1.0,
            step=0.01,
            value=1.0,
        )

        if "opencv_image" not in st.session_state:
            st.session_state.raw_image_gray = None

        # Reading image if it has been uploaded
        if uploaded_image is not None:
            st.session_state.raw_image_gray = read_image_to_opencv(
                uploaded_image, rescale_factor
            )

        submit_button = st.form_submit_button("Analyze image")

    col1a, col1b = st.columns(2)
    with col1a:
        st.write("Input image (grayscale)")
        if st.session_state.raw_image_gray is not None:
            st.image(st.session_state.raw_image_gray, use_container_width=True)
            st.text("Image shape: " + str(st.session_state.raw_image_gray.shape))

    with col1b:
        with st.form("form2", enter_to_submit=False, clear_on_submit=False):
            # coherence gamma, hist blur, inner/outer sigma, epsilon, num bins
            collect_params_from_user()
            submit_button_2 = st.form_submit_button("Confirm options")


# col2 should be for visualizing processed images, and should have everything update live.
# Add dropdown menu for which layers to view: intensity, angle, and coherence - done
with col2:
    imageToDisplay = st.selectbox(
        "Image to display:",
        (
            "Intensity, Coherence, and Angle",
            "Coherence and Angle only",
            "Coherence only",
            "Angle only",
        ),
    )

    if st.session_state.raw_image_gray is not None:
        raw_image_gray = st.session_state.raw_image_gray

        coh, ang = coh_ang_calc(
            raw_image_gray,
            st.session_state.sigma_to_ydim_ratio,
            st.session_state.innerSigma_to_ydim_ratio,
            st.session_state.epsilon
            / raw_image_gray.shape[0]
            / raw_image_gray.shape[0],
        )  # this is the important bit

        coh_gammaified = np.power(coh, st.session_state.coherence_gamma)
        weighted_hist = weightedHistogram(
            coh_gammaified, ang, st.session_state.num_bins
        )
        blurred_histogram = (
            gaussian_filter1d(
                weighted_hist, sigma=st.session_state.histogram_blur_sigma, mode="wrap"
            )
            if st.session_state.histogram_blur_sigma != 0
            else weighted_hist
        )

        blockreduce = st.number_input(
            "choose k for (k by k) block reduce",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
        )

        display_selected_falsecolor(
            raw_image_gray, coh, ang, mode=imageToDisplay, blockreduce=blockreduce
        )
        st.text(
            "outer sigma in pixels: "
            + str(st.session_state.sigma_to_ydim_ratio * raw_image_gray.shape[0])
            + (" (should be at least .5px)")
        )

        plot_angle_histogram(blurred_histogram, st.session_state.histogram_blur_sigma)
        file_name = st.text_input(
            "file name for download", value="unnamed_angle_histogram"
        )
        st.download_button(
            "Download " + file_name + ".csv",
            pd.DataFrame(blurred_histogram).to_csv(index=False).encode("utf-8"),
            file_name + ".csv",
            "test/csv",
            key="download-csv",
        )
    else:
        st.write('No image uploaded yet - click "Analyze"?')
