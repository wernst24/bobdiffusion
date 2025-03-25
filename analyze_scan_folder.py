from modules import coh_ang_calc
import os
import cv2 as cv
from skimage import color
# import numpy as np

# print(os.listdir())
# os.chdir("synth_scans")
# print(os.listdir())

scan_path = "synth_scans"
# analysis_folder = "bta/"
os.chdir(scan_path)
# if analysis_folder not in os.listdir():
    # os.mkdir(analysis_folder)

default_settings = input("default settings [y/n]: ")
if default_settings == "y":
    sigma_outer = 0.005
    sigma_inner = 0.005
    epsilon = 0.0001
else:
    sigma_outer = float(input("outer sigma: "))
    sigma_inner = float(input("inner sigma: "))
    epsilon = float(input("epsilon: "))




for i, file in enumerate(os.listdir()):
    print(f"File {i}: {file} \t read")
    coh_filepath = "coh_" + file
    ang_filepath = "ang_" + file
    # print(cv.imread(file))
    img = color.rgb2gray(cv.imread(file))
    
    coh, ang = coh_ang_calc(img, sigma_outer, sigma_inner, epsilon)
    print(coh.dtype)
    # print(f'img min: {img.min()}, img max: {img.max()}')
    cv.imwrite(coh_filepath, coh[:-4] + ".jpg")
    cv.imwrite(ang_filepath, ang[:-4] + ".jpg")
    
    print(f"File {i}: {file} \t coherence & angle written")
    # img = rescale(img, 0.1)
    

    # print(f'coh min: {coh.min()}, coh max: {coh.max()}')
    
    while True:
        # cv.imshow("Image " + str(i), cv.hconcat([img, coh*255, ang + np.pi]))
        cv.imshow('imgs', img)
        cv.imshow('coh', coh)
        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()
    pass