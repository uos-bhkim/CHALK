from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import dl_module # TO-BE : Changed to SHM Model
from shutil import copyfile

import cv2
import numpy as np

from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

app = Flask(__name__, static_url_path='/static')
config_file = '/home/user/UOS-SSaS Dropbox/05. Data/03. Checkpoints/2021.07.22_deeplabv3plus_r50-d8_769x769_40k_concrete_crack_cs_xt/deeplabv3_r101-d8_769x769_40k_cityscapes.py'
checkpoint_file = '/home/user/UOS-SSaS Dropbox/05. Data/03. Checkpoints/2021.07.22_deeplabv3plus_r50-d8_769x769_40k_concrete_crack_cs_xt/iter_40000.pth'
# build the model from a config file and a checkpoint file
segmentor = init_segmentor(config_file, checkpoint_file, device='cuda:1')

@app.route("/")
def index():
	return render_template("home.html")

@app.route("/labeling")
def labeling():

	if request.method == 'POST':
        # do stuff when the form is submitted

        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
		return redirect(url_for('index'))

	return render_template("labeling-page.html")

@app.route("/upload", methods=['POST'])
def upload_file():
	"""
	1. Upload View
	"""
	f = request.files['file']
	# 저장할 경로 + 파일명
	upload_path = "static/images/upload/" + secure_filename(f.filename) # 저장 첫번째

	f.save(upload_path)
	
	filename_png = secure_filename(f.filename).split('.')[0] + '.png'

	original_img_path = "static/images/original/" + filename_png
	colored_img_path = "static/images/colored_img/" + filename_png # 저장 첫번째
	label_img_path = "static/images/label/" + filename_png # 저장 첫번째
	
	upload_img = cv2.imread(upload_path)
	label_img = np.zeros((upload_img.shape[0], upload_img.shape[1]), dtype = np.uint8 )

	cv2.imwrite(original_img_path, upload_img)
	cv2.imwrite(colored_img_path, upload_img)
	cv2.imwrite(label_img_path, label_img)

	return redirect(url_for("crop_file", file_name=filename_png))

@app.route("/crop/<file_name>", methods=['GET', 'POST'])
def crop_file(file_name: str = None):
	"""
	2. Crop View
	"""
	if request.method == 'POST':
		res = request.get_json()
		name = file_name
		arr = res['pos']

		path = "static/images/" + name
		model_res = dl_module.model(name, arr, segmentor) # cropped_origin, label, colored_cropped_origin

		cropped_origin = model_res[0]
		cropped_label = model_res[1]
		colored_cropped_origin = model_res[2]
		print(colored_cropped_origin)

		return redirect(url_for("manipulate_file", file_name=colored_cropped_origin))

	return render_template("crop.html", file_name=file_name)

@app.route("/manipulate/<file_name>")
def manipulate_file(file_name: str = None):
	"""
	3. Manipulate View
	"""
	return render_template("manipulate.html", file_name=file_name)

# @app.route("/cropped/<file_name>", methods=["POST"])
# def reload(file_name: str, pos: list[int]):

@app.route("/done/<file_name>", methods=["POST"])
def return_to_image(file_name: str):
	"""
	변경 완료 버튼을 눌렀을 때 반응할 함수
	"""
	return redirect(url_for("crop_file", file_name=secure_filename(file_name)))
	#  render_template("crop.html", file_name=file_name)

@app.route("/done_img/<file_name>", methods=["POST"])
def return_to_home(file_name: str):
	"""
	완료 후 홈으로 귀환
	"""
	return redirect(url_for("index"))
	#  render_template("crop.html", file_name=file_name)


@app.route('/save/<file_name>')
def save_image(file_name: str):
	file_path = "static/images/final/" + file_name
	return send_file(file_path,
					mimetype='image/jpg',
					attachment_filename="example_final.jpeg",
					as_attachment=True)




if __name__ == "__main__":

	app.run(host='172.16.113.179', port = 8080, debug=True)