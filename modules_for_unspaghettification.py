from cv2 import imdecode
from numpy import asarray, uint8
from skimage.transform import resize
from skimage.color import rgb2gray

def read_image_to_opencv(uploaded_image, rescale_factor):
    image_bytes = asarray(bytearray(uploaded_image.read()), dtype=uint8)
    cv_image = imdecode(image_bytes, 1)
    cv_image_gray = rgb2gray(cv_image)
    # return transform.rescale(cv_image_gray, rescale_factor, anti_aliasing=True)
    return resize(cv_image_gray, (200, 200))