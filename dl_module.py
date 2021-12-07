import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import os.path as osp
import time 
from inference import inference_segmentor_sliding_window
from mmseg.core.evaluation import get_palette

# has an error : reload()
# from model.inference import run_model

def model(img_path: "str", pos: list, segmentor ) -> 'numpy.ndarray':
    """
    Crop and Label image
    pos: [x, y, w, h]
    """
    # 원본 이미지 path를 구성하고, 이를 crop
    origin_path = "static/images/original/" + img_path
    colored_img_path = "static/images/colored_img/" + img_path
    cropped_colored_img_path = "static/images/cropped/colored/" + img_path
    label_path = "static/images/label/" + img_path

    origin_img = cv2.imread(origin_path)
    colored_img = cv2.imread(colored_img_path)
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

    subset_origin_img = crop_image(origin_img, pos)

    img_path_filename_png = img_path.split('.')[0] + '.png'
    
    colored_subset, label_subset = inference_segmentor_sliding_window(segmentor, subset_origin_img, get_palette('concrete_crack_as_cityscapes')[1:], 1)
    # cv2.imwrite(label_path, mask_output)

    updated_colored_img = update_colored_img(colored_img, colored_subset, pos)
    updated_label = update_label(label, label_subset, pos)


    cv2.imwrite(cropped_colored_img_path, colored_subset)
    cv2.imwrite(colored_img_path, updated_colored_img)
    cv2.imwrite(label_path, updated_label)

    # convert_original_to_labeled(subset_origin_img, label_img, img_path_filename_png)
    
    return (subset_origin_img, label_subset, img_path_filename_png)

def convert_pos_to_idx(pos: list) -> tuple:

    x, y, w, h = pos
    
    sx, sy, ex, ey = x, y, x+w, y+h
    if w < 0: 
        sx, ex = ex, sx
    if h < 0: 
        sy, ey = ey, sy

    return sx, sy, ex, ey


def update_label(label: 'numpy.ndarray', label_subset: 'numpy.ndarray', pos: list) -> 'numpy.ndarray':

    update_label = copy.deepcopy(label)
    
    sx, sy, ex, ey = convert_pos_to_idx(pos)

    update_label[sy:ey, sx:ex] = np.squeeze(label_subset)

    return update_label



def update_colored_img(colored_img: 'numpy.ndarray', colored_subset: 'numpy.ndarray', pos: list) -> 'numpy.ndarray':

    recolored_img = copy.deepcopy(colored_img)
    
    sx, sy, ex, ey = convert_pos_to_idx(pos)

    recolored_img[sy:ey, sx:ex, :] = colored_subset

    return recolored_img

def crop_image(target: 'numpy.ndarray', pos: list) -> 'numpy.ndarray':

    tmp = copy.deepcopy(target)

    sx, sy, ex, ey = convert_pos_to_idx(pos)
    
    return tmp[sy:ey, sx:ex]



def convert_original_to_labeled(origin: 'numpy.ndarray', label: 'numpy.ndarray', img_filename: "str") -> 'numpy.ndarray':
    """
    It should be guaranteed that width, height of parameters are same
    """
    tmp = copy.deepcopy(origin)

    row = len(origin)
    col = len(origin[0])
    
    for i in range(row):
        for j in range(col):
            if label[i][j] == 1:
                tmp[i][j] = [0, 0, 255] # BGR, Current color is RED

    tmp_filename =  img_filename
    tmp_path = "static/images/temp/colored_img/" + tmp_filename
    cv2.imwrite(tmp_path, tmp)

    
