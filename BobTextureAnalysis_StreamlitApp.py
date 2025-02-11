import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage import color
from skimage.transform import rescale
import pandas as pd
from matplotlib.backends.backend_agg import RendererAgg
from scipy.ndimage import gaussian_filter1d

sys.dont_write_bytecode = True
sys.tracebacklimit = 0
from modules import *

st.set_page_config(
    page_title="BobHistogram",
    page_icon="🔬",
    layout="wide",
)

col1, col2 = st.columns(2)

# col1 should be for uploading image only: upload image, choose downscaling factor, and then preview at bottom.
with col1:
    # title for form
    st.markdown("# weighted histogram")

    # Everything in this block will wait until submitted - it should contain uploading the images and infrequently changed parameters - initial downscale, etc.
    with st.form("form1", enter_to_submit=False, clear_on_submit=False):
        msg = "Upload a 2D image to be analyzed. Downsizing is reccomended for larger images"


        uploaded_image = st.file_uploader(msg, type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=False)

        # TODO: take this out of the form, because inverting the images doesn't (shouldn't) change the coherence or angle calculation, but changing
        # if inverted forces recalculation
        # Change to parameter for orient_hsv?

        # No idea what label will make sense for this
        rescale_factor = st.number_input("Lazy scale space (downscale)", min_value=0.01, max_value=1.0, step=0.01, value=1.0)

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
        with st.form("form2", enter_to_submit = False, clear_on_submit = False):
            st.write("Processing options")

            # These don't need checking for NaN, because they have default values
            # st.session_state.ksize = st.number_input(value=1, min_value=1, max_value=51, step=2, label="kernel diameter")

            st.session_state.histogram_blur_sigma = st.number_input(value=1.0, min_value=.01, max_value=100.0, step=.1, label="histogram blur sigma")

            st.session_state.innerSigma = st.number_input(value=1.0, min_value=.01, max_value=10.0, step=.1, label="inner sigma")

            st.session_state.epsilon = st.number_input(value=1e-8, min_value=1e-8, max_value=1.0, step=.01, label="epsilon")

            st.session_state.num_bins = st.number_input(value=100, min_value=100, max_value=1000, step=1, label="num_bins")
            submit_button_2 = st.form_submit_button("Confirm options")

# col2 should be for visualizing processed images, and should have everything update live.
# Add dropdown menu for which layers to view: intensity, angle, and coherence - done
with col2:
    _lock = RendererAgg.lock
    
    
    
    if st.session_state.raw_image_gray is not None:
        raw_image_gray = st.session_state.raw_image_gray
        coh, ang = coh_ang_calc(raw_image_gray, st.session_state.innerSigma, st.session_state.epsilon)
        weighted_hist = weightedHistogram(coh, ang, st.session_state.num_bins)
        blurred = gaussian_filter1d(weighted_hist, sigma=st.session_state.histogram_blur_sigma ,mode='wrap')

        with _lock:
            
            fig, ax = plt.subplots()
            ax.bar(range(st.session_state.num_bins), weighted_hist, width=1, color="gray")
            ax.set_title("Coherence-Weighted Histogram of Angles")
            ax.set_ylabel("Sum of coherence (roughly counts)")
            ax.set_xlabel("Bin number (roughly angle) - 0 is vertical, increasing CW")
            ax.plot(range(st.session_state.num_bins), blurred, c='black')
            # ax.set_xticks()
            st.pyplot(fig)
        
        st.text((ang.min(), ang.max()))
        st.image(orient_hsv(st.session_state.raw_image_gray, coh, ang, mode='angle'), use_container_width=True)

        st.download_button(
            "Download weighted histogram as csv",
            pd.DataFrame(weighted_hist).to_csv(index=False).encode('utf-8'),
            "weighted_histogram.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.write("No image uploaded yet - click \"Analyze\"?")