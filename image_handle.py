from PIL import Image, ImageFilter
import numpy as np
import cv2
from json import dumps

# !-- CV2는 BGR 순으로 Color Channel을 구성합니다. (08/28 fixed)
# original 이미지 넣으면, label img 받아와서 rendering

def count_in_list(lst: list) -> int:
	"""
	Function that should be counted in list
	"""
	cnt = 0
	for i in range(len(lst)):
		for j in range(len(lst[0])):
			if lst[i][j] == 1:
				cnt += 1
	return cnt

def color_check(origin_color: list, target_color: list) -> bool:
	"""
	Function that check if it is the color we're looking for in C.I.
	Set C.I. properly for handling between accuracy and robustness.
	"""
	CI = 20 # Confidential Interval

	if target_color[0]-CI <= origin_color[0] <= target_color[0]+CI\
	and target_color[1]-CI <= origin_color[1] <= target_color[1]+CI\
	and target_color[2]-CI <= origin_color[2] <= target_color[2]+CI:
		return True
	else:
		return False

def image_to_array(img: str) -> 'numpy.ndarray':
	"""
	Function that converts image to numpy array
	"""
	im = cv2.imread(img)
	print(type(im))
	return im

def image_manipulate_pixel(img: 'numpy.ndarray', color: list[int], point: tuple[int]) -> 'numpy.ndarray':
	"""
	Function that manipulate pixel color to `color`
	"""
	# 1. 특정 픽셀 부분 assignment로 수정하기
	# 2. 다 한다음 그 ndarray를 반환한다음
	# 3. 그거를 image화해서...!
	for i, j in	point:
		img[i][j] = color
	return img

def find_label(img: 'numpy.ndarray', color: list[int]) -> list: # TODO: fix to ndarray
	"""
	Function that find same pixel in image and return it's coordinate
	"""
	res = []
	for i in range(256):
		for j in range(256):
			if color_check(img[i][j], color):
				res.append((i, j)) 
	return res

def find_label_and_convert_to_image(img: 'numpy.ndarray', color: list[int]) -> 'Image': # TODO: fix to ndarray
	"""
	Function that find same pixel in image and return it's coordinate
	"""
	res = []
	new_ndarray = np.zeros(shape=(256, 256), dtype=int)
	for i in range(256):
		for j in range(256):
			if color_check(img[i][j], color):
				new_ndarray[i][j] = 1
	new_img = Image.fromarray(new_ndarray, 'L') 
	print(new_ndarray.shape)
	print(count_in_list(new_ndarray))
	return new_img

im = image_to_array("labeled_test.jpeg")
new_image = find_label_and_convert_to_image(im, [40, 40, 160])
new_image.save("test.png", "PNG")
