from modules import coh_ang_calc, weightedHistogram, orient_hsv
from modules_for_unspaghettification import read_image_to_opencv, plot_angle_histogram
import streamlit as st
import numpy as np

import sys

sys.dont_write_bytecode = True
sys.tracebacklimit = 0


st.set_page_config(
    page_title="BobTextureAnalysis",
    page_icon="🔬",
    layout="wide",
)

col1, col2 = st.columns(2)

if "synthetic_drumhead" not in st.session_state:
    st.session_state.synthetic_drumhead = 0
verdict1 = 0
verdict2 = 0
if st.session_state.synthetic_drumhead != 0:
    verdict1 = "Synthetic" if st.session_state.synthetic_drumhead == 1 else "Natural"
    verdict2 = "Synthetic" if st.session_state.synthetic_drumhead == 2 else "Natural"

with col1:
    st.markdown("# Synthetic/Natural Comparison\nUpload one image each of a natural and synthetic drumhead. Make sure the sample covers the entire frame; edges will mess with the algorithm.")
    col1a, col1b = st.columns(2)
    with col1a:
        # st.markdown("# weighted histogram")

        with st.form("form1", enter_to_submit=False, clear_on_submit=False):
            # upload and get rescale factor
            msg = "Upload first image for comparison, and click \"Analyze image\""
            uploaded_image1 = st.file_uploader(
                msg, type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=False
            )

            if "opencv_image1" not in st.session_state:
                st.session_state.raw_image_gray1 = np.zeros((100, 100))

            # Reading image if it has been uploaded
            if uploaded_image1 is not None:
                st.session_state.raw_image_gray1 = read_image_to_opencv(
                    uploaded_image1, 0.1
                )

            submit_button = st.form_submit_button("Analyze image")
        if (st.session_state.synthetic_drumhead != 0):
            st.write("Image 1: " + verdict1)

    with col1b:
        with st.form("form2", enter_to_submit=False, clear_on_submit=False):
            # upload and get rescale factor
            msg = "Upload second image for comparison, and click \"Analyze image\""
            uploaded_image2 = st.file_uploader(
                msg, type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=False
            )

            if "opencv_image2" not in st.session_state:
                st.session_state.raw_image_gray2 = np.zeros((100, 100))

            # Reading image if it has been uploaded
            if uploaded_image2 is not None:
                st.session_state.raw_image_gray2 = read_image_to_opencv(
                    uploaded_image2, 0.1
                )

            submit_button = st.form_submit_button("Analyze image")
        if (st.session_state.synthetic_drumhead != 0):
            st.write("Image 2: " + verdict2)

    # st.write("Input image (grayscale)")
    # if st.session_state.raw_image_gray is not None:
    #     st.image(st.session_state.raw_image_gray, use_container_width=False)
    # st.write(f"Shape: {st.session_state.raw_image_gray.shape}")

with col2:

    if st.session_state.raw_image_gray1 is not None:
        raw_image_gray1 = st.session_state.raw_image_gray1
        raw_image_gray2 = st.session_state.raw_image_gray2

        coh1, ang1 = coh_ang_calc(
            raw_image_gray1,
            0.008,
            0.015,
            1.0
            / raw_image_gray1.shape[0]
            / raw_image_gray1.shape[0],
        )  # this is the important bit

        coh2, ang2 = coh_ang_calc(
            raw_image_gray2,
            0.008,
            0.015,
            1.0
            / raw_image_gray2.shape[0]
            / raw_image_gray2.shape[0],
        )  # this is the important bit

        num_bins = 200

        weighted_hist1 = weightedHistogram(coh1, ang1, num_bins)
        weighted_hist2 = weightedHistogram(coh2, ang2, num_bins)

        normalized_hist1 = weighted_hist1 / weighted_hist1.sum()
        normalized_hist2 = weighted_hist2 / weighted_hist2.sum()
        varx1 = 0.0
        vary1 = 0.0
        varx2 = 0.0
        vary2 = 0.0 
        
        for i in range(num_bins):
            varx1 += np.cos(i * 2 * np.pi / num_bins) * normalized_hist1[i]
            vary1 += np.sin(i * 2 * np.pi / num_bins) * normalized_hist1[i]
            varx2 += np.cos(i * 2 * np.pi / num_bins) * normalized_hist2[i]
            vary2 += np.sin(i * 2 * np.pi / num_bins) * normalized_hist2[i]
        var1 = 1 - np.sqrt(varx1 ** 2 + vary1 ** 2)
        var2 = 1 - np.sqrt(varx2 ** 2 + vary2 ** 2)
        metric1 = var1 * weighted_hist1.sum()
        metric2 = var2 * weighted_hist2.sum()
        
        st.session_state.synthetic_drumhead = 1 if metric1 > metric2 else 2
        col2a, col2b = st.columns(2)
        with col2a:
            # st.write(f"metric 1: {metric1:.2f}")
            # st.write(f"variance 1: {var1:.2f}")
            st.write("Image 1")
            st.image(
            orient_hsv(raw_image_gray1, coh1, ang1, mode="all"),
            use_container_width=True,
            )
            # plot_angle_histogram(weighted_hist1, 0)

        with col2b:
            # st.write(f"metric 2: {metric2:.2f}")
            # st.write(f"variance 2: {var2:.2f}")
            st.write("Image 2")
            st.image(
            orient_hsv(raw_image_gray2, coh2, ang2, mode="all"),
            use_container_width=True,
            )
            # plot_angle_histogram(weighted_hist2, 0)
