from modules import coh_ang_calc, weightedHistogram, orient_hsv
from modules_for_unspaghettification import read_image_to_opencv
import streamlit as st
import numpy as np

import sys

sys.dont_write_bytecode = True
sys.tracebacklimit = 0


st.set_page_config(
    page_title="BobHistogram",
    page_icon="🔬",
    layout="wide",
)

col1, col2 = st.columns(2)

with col1:
    # st.markdown("# weighted histogram")

    with st.form("form1", enter_to_submit=False, clear_on_submit=False):
        # upload and get rescale factor
        msg = "Upload an image for analysis, or take a photo."
        uploaded_image = st.file_uploader(
            msg, type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=False
        )

        if "opencv_image" not in st.session_state:
            st.session_state.raw_image_gray = np.zeros((100, 100))

        rescale_factor = 1
        # Reading image if it has been uploaded
        if uploaded_image is not None:
            st.session_state.raw_image_gray = read_image_to_opencv(
                uploaded_image, 0.1
            )

        submit_button = st.form_submit_button("Analyze image")

    st.write("Input image (grayscale)")
    if st.session_state.raw_image_gray is not None:
        st.image(st.session_state.raw_image_gray, use_container_width=False)
    st.write(f"Shape: {st.session_state.raw_image_gray.shape}")

with col2:

    if st.session_state.raw_image_gray is not None:
        raw_image_gray = st.session_state.raw_image_gray

        coh, ang = coh_ang_calc(
            raw_image_gray,
            0.008,
            0.015,
            1.0
            / raw_image_gray.shape[0]
            / raw_image_gray.shape[0],
        )  # this is the important bit

        weighted_hist = weightedHistogram(coh, ang, 200)
        blurred_histogram = weighted_hist

        st.image(
        orient_hsv(raw_image_gray, coh, ang, mode="all"),
        use_container_width=True,
        )