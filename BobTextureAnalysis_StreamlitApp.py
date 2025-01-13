import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import time
import sys
# from functools import partial
# import pandas as pd

sys.dont_write_bytecode = True
sys.tracebacklimit = 0
from modules import *

st.set_page_config(
    page_title="evil BobTextureAnalysis",
    page_icon="🔬",
    layout="wide",
)

col1, col2 = st.columns(2)

# col1 should be for uploading image only: upload image, choose downscaling factor, and then preview at bottom.
with col1:
    # title for form
    st.markdown("# filter testing")

    # Everything in this block will wait until submitted - it should contain uploading the images and infrequently changed parameters - initial downscale, etc.
    with st.form("form1", enter_to_submit=False, clear_on_submit=False):
        msg = "Upload a 2D image to be analyzed. Downsizing is reccomended for larger images"


        uploaded_image = st.file_uploader(msg, type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=False)

        # TODO: take this out of the form, because inverting the images doesn't (shouldn't) change the coherence or angle calculation, but changing
        # if inverted forces recalculation
        # Change to parameter for orient_hsv?

        # No idea what label will make sense for this
        rescale_factor = st.number_input("Downscale factor (1 for no downscale - 0.01 for 100x smaller)", min_value=0.01, max_value=1.0, step=0.01, value=1.0)

        # Image
        if "opencv_image" not in st.session_state:
            st.session_state.raw_image_gray = None

        # Reading image if it has been uploaded
        if uploaded_image is not None:
            image_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            cv_image = cv.imdecode(image_bytes, 1)
            cv_image_gray = color.rgb2gray(cv_image) # Convert to grayscale
            
            # rescale with skimage
            cv_image_rescaled = rescale(cv_image_gray, rescale_factor, anti_aliasing=True)

            st.session_state.raw_image_gray = cv_image_rescaled
        
        submit_button = st.form_submit_button("Analyze image")
    
    col1a, col1b = st.columns(2) # col1a is for displaying image, 1b is for parameters
    with col1a:
        st.write("Input image (grayscale)")
        if st.session_state.raw_image_gray is not None:
            st.image(st.session_state.raw_image_gray, use_container_width=True)
    
    with col1b:
        st.write("Processing options")

        # These don't need checking for NaN, because they have default values
        st.session_state.ksize = st.number_input(value=1, min_value=1, max_value=9, step=2, label="kernel diameter")

        st.session_state.spaceSigma = st.number_input(value=1, min_value=1, max_value=100, step=1, label="space sigma")

        st.session_state.colorSigma = st.number_input(value=1, min_value=1, max_value=100, step=1, label="color sigma")

# col2 should be for visualizing processed images, and should have everything update live.
# Add dropdown menu for which layers to view: intensity, angle, and coherence - done
with col2:
    # Selection for which image to view
    imageToDisplay = st.selectbox("Image to display:", ("Default image", "Gaussian filter", "Bilateral filter"))
    if st.session_state.raw_image_gray is not None:
        raw_image_gray = st.session_state.raw_image_gray
        gfilt = cv.GaussianBlur(raw_image_gray, (st.session_state.ksize, st.session_state.ksize), st.session_state.spaceSigma)
        bfilt = cv.bilateralFilter(raw_image_gray.astype(np.float32), st.session_state.ksize, st.session_state.colorSigma, st.session_state.spaceSigma)
        # # calculate coherence and angle at a given sigma inner scale
        # coherence, two_phi = coh_ang_calc(raw_image_gray, sigma_inner=st.session_state.inner_sigma, epsilon=st.session_state.epsilon)
        # two_phi *= -1 # flip direction of increasing angles to CCW

        
    
    # Display image based on user selection
    if st.session_state.raw_image_gray is not None:
        if imageToDisplay == "Default image":
            image_to_show = raw_image_gray
        elif imageToDisplay == "Gaussian filter":
            image_to_show = gfilt
        elif imageToDisplay == "Bilateral filter":
            image_to_show = bfilt
        else:
            assert False, "image to display selector somehow went horribly wrong"
        st.image(image_to_show, use_container_width=True)
    else:
        st.write("No image uploaded yet - click \"Analyze\"?")