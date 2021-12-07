#%%
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import os
import numpy as np
import cv2
import slidingwindow as sw
import tqdm
import glob

config_file = '/home/user/UOS-SSaS Dropbox/05. Data/03. Checkpoints/2021.07.22_deeplabv3plus_r50-d8_769x769_40k_concrete_crack_cs_xt/deeplabv3_r101-d8_769x769_40k_cityscapes.py'
checkpoint_file = '/home/user/UOS-SSaS Dropbox/05. Data/03. Checkpoints/2021.07.22_deeplabv3plus_r50-d8_769x769_40k_concrete_crack_cs_xt/iter_40000.pth'
# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:1')

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        imageBGR = cv2.imdecode(n, flags)
        return cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(e)
        return None


def imwrite(filename, imageRGB, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        imageBGR = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2BGR)
        result, n = cv2.imencode(ext, imageBGR, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
                return True
        else:
                return False

    except Exception as e:
        print(e)
        return False

def inference_segmentor_sliding_window(model, input_img, color_mask, num_classes, 
                                      score_thr = 0.1, window_size = 1024, overlap_ratio = 0.1,):


    '''
    :param model: is a mmdetection model object
    :param input_img : str or numpy array
                    if str, run imread from input_img
    :param score_thr: is float number between 0 and 1.
                   Bounding boxes with a confidence higher than score_thr will be displayed,
                   in 'img_result' and 'mask_output'.
    :param window_size: is a subset size to be detected at a time.
                        default = 1024, integer number
    :param overlap_ratio: is a overlap size.
                        If you overlap sliding windows by 50%, overlap_ratio is 0.5.

    :return: img_result
    :return: mask_output

    '''

    # color mask has to be updated for multiple-class object detection
    if isinstance(input_img, str) :
        img = imread(input_img)
    else :
        img = input_img

    # Generate the set of windows, with a 256-pixel max window size and 50% overlap
    windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, window_size, overlap_ratio)
    mask_output = np.zeros((img.shape[0], img.shape[1], num_classes), dtype=np.uint8)

#     if isinstance(input_img, str) :
#         tqdm_window = tqdm(windows, ascii=True, desc='inference by sliding window on ' + os.path.basename(input_img))
#     else :
#         tqdm_window = tqdm(windows, ascii=True, desc='inference by sliding window ')

    for window in windows :
        # Add print option for sliding window detection
        img_subset = img[window.indices()]
        results = inference_segmentor(model, img_subset)[0]
        results_onehot = (np.arange(num_classes) == results[...,None]-1).astype(int)
        
        mask_output[window.indices()] = mask_output[window.indices()] + results_onehot

    mask_output[mask_output > 1] = 1

    mask_output_bool = mask_output.astype(np.bool)

    # Add colors to detection result on img
    img_result = img
    for num in range(num_classes) : 
        img_result[mask_output_bool[:,:,num-1], :] = img_result[mask_output_bool[:,:,num-1],:] * 0.01 + np.asarray(color_mask[num-1], dtype = float) * 0.99
        print(num)
        print(color_mask[num-1])

    return img_result, mask_output

def run_model():
    img_folder = '/home/user/ssi_proj/static/images/cropped'
    img_temp_folder = '/home/user/ssi_proj/static/images/temp'

    img_list = glob.glob(os.path.join(img_folder, '*.png')) 

    for img_path in img_list : 
        img_subset = imread(img_path)
        img_filename = img_path.split('/')[-1]
        img_save_path = os.path.join(img_temp_folder, img_filename)

        _, mask_output = inference_segmentor_sliding_window(model, img_subset, get_palette('concrete_crack_as_cityscapes')[1:], 1)
        cv2.imwrite(img_save_path, mask_output)


if __name__ == "__main__" :

    while True :
        run_model()
