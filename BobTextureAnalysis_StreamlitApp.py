import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("agg")  # Use non-GUI backend
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
# nevermind
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

            #TODO: add crop

            st.session_state.raw_image_gray = cv_image_rescaled
            # maybe make dtype f16 instead of f64
        
        submit_button = st.form_submit_button("Analyze image")
    
    col1a, col1b = st.columns(2) # col1a is for displaying image, 1b is for parameters
    with col1a:
        st.write("Input image (grayscale)")
        if st.session_state.raw_image_gray is not None:
            st.image(st.session_state.raw_image_gray, use_container_width=True)
            st.text("Image shape: " + str(st.session_state.raw_image_gray.shape))
    
    with col1b:
        with st.form("form2", enter_to_submit = False, clear_on_submit = False):
            st.write("Processing options")

            # These don't need checking for NaN, because they have default values
            # st.session_state.ksize = st.number_input(value=1, min_value=1, max_value=51, step=2, label="kernel diameter")

            st.text("Sorry for unintutitve parameters-currently working on it")

            st.session_state.coherence_gamma = st.number_input(value=1.0, min_value=.5, max_value=4.0, step=0.0, label="coherence gamma", format="%.6f")

            st.session_state.histogram_blur_sigma = st.number_input(value=0.0, min_value=0.0, max_value=100.0, step=0.0, label="histogram blur sigma", format="%.6f")

            # st.markdown("this bit is for outer sigma - know image dimensions. Sigma for convolution is calculated with \"sigma = image.shape[0] * sigma_to_ydim_ratio\". Calculation *should* be (sigma [m])/(image ydim [m])")

            st.session_state.sigma_to_ydim_ratio = st.number_input(value=0.001, min_value=0.0000001, max_value=100.0, step=0.0, label="sigma to ydim ratio", format="%.6f")

            st.session_state.innerSigma = st.number_input(value=1.0, min_value=.01, max_value=10.0, step=0.0, label="inner sigma", format="%.6f")

            st.session_state.epsilon = st.number_input(value=1e-8, min_value=1e-8, max_value=1.0, step=-1.0, label="epsilon", format="%.6f")

            st.session_state.num_bins = st.number_input(value=180, min_value=180, max_value=5000, step=0, label="num_bins")
            submit_button_2 = st.form_submit_button("Confirm options")

    # still col 1
    

# col2 should be for visualizing processed images, and should have everything update live.
# Add dropdown menu for which layers to view: intensity, angle, and coherence - done
with col2:

    imageToDisplay = st.selectbox("Image to display:", ("Intensity, Coherence, and Angle", "Coherence and Angle only", "Coherence only", "Angle only (black & white)"))
    
    if st.session_state.raw_image_gray is not None:
        raw_image_gray = st.session_state.raw_image_gray
        # st.markdown("before changed bit")
        coh, ang = coh_ang_calc(raw_image_gray, st.session_state.sigma_to_ydim_ratio, st.session_state.innerSigma, st.session_state.epsilon) # this is the important bit
        # st.markdown("after changed bit")
        coh_gammaified = np.power(coh, st.session_state.coherence_gamma)
        weighted_hist = weightedHistogram(coh_gammaified, ang, st.session_state.num_bins)
        blurred = gaussian_filter1d(weighted_hist, sigma=st.session_state.histogram_blur_sigma ,mode='wrap') if st.session_state.histogram_blur_sigma != 0 else weighted_hist

        fig, ax = plt.subplots()
        ax.bar(np.linspace(0, 180, st.session_state.num_bins), blurred, width=1, color=matplotlib.colormaps['hsv'](np.linspace(0, 1, st.session_state.num_bins)))
        ax.set_title("Coherence-Weighted Histogram of Angles")
        ax.set_ylabel("Sum of coherence (roughly counts)")
        ax.set_xlabel("Angle [deg] - 0 vertical, increasing CCW")
        # ax.plot(np.linspace(0, 180, st.session_state.num_bins), blurred, c='black', alpha = 0.5)
        # ax.set_xticks()
        st.pyplot(fig)
        
        # st.text((ang.min(), ang.max()))
        all_img = orient_hsv(raw_image_gray, coh, ang, mode='all')
        ang_img = orient_hsv(raw_image_gray, coh, ang, mode='angle')
        coh_img = orient_hsv(raw_image_gray, coh, ang, mode='coherence')
        ang_img_bw = orient_hsv(raw_image_gray, coh, ang, mode='angle_bw')
        # st.text(coh.max())

        st.text("outer sigma in pixels: " + str(st.session_state.sigma_to_ydim_ratio * raw_image_gray.shape[0]) + (" (should be at least .5px)"))

        if imageToDisplay == "Intensity, Coherence, and Angle":
            image_to_show = all_img
        elif imageToDisplay == "Coherence and Angle only":
            image_to_show = ang_img
        elif imageToDisplay == "Coherence only":
            image_to_show = coh_img
        else:
            image_to_show = ang_img_bw
        st.image(image_to_show, use_container_width=True)

        
        k = st.number_input("choose k for (k by k) block reduce", min_value=1, max_value=100, value=1, step=1)
        (h, w) = raw_image_gray.shape
        raw_image_gray_small = raw_image_gray[:h//k * k, :w//k * k].reshape(h//k, k, w//k, k).mean(axis=(1, 3))
        coherence_small, two_phi_small = downscale_coh_ang(coh, ang, k)

        all_img_small = orient_hsv(raw_image_gray_small, coherence_small, two_phi_small, mode='all')
        coh_img_small = orient_hsv(raw_image_gray_small, coherence_small, two_phi_small, mode='coherence')
        ang_img_small = orient_hsv(raw_image_gray_small, coherence_small, two_phi_small, mode='angle')
        ang_img_bw_small = orient_hsv(raw_image_gray_small, coherence_small, two_phi_small, mode="angle_bw")

        if imageToDisplay == "Intensity, Coherence, and Angle":
            image_to_show2 = all_img_small
        elif imageToDisplay == "Coherence and Angle only":
            image_to_show2 = ang_img_small
        elif imageToDisplay == "Coherence only":
            image_to_show2 = coh_img_small
        else:
            image_to_show2 = ang_img_bw_small
        st.image(image_to_show2, use_container_width=False)


        file_name = st.text_input("file name for download", value="unnamed_angle_histogram")

        st.download_button(
            "Download " + file_name + ".csv",
            pd.DataFrame(blurred).to_csv(index=False).encode('utf-8'),
            file_name + ".csv",
            "test/csv",
            key='download-csv'
        )
    else:
        st.write("No image uploaded yet - click \"Analyze\"?")