# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:36:02 2020

@author: Colin Cumming
"""
import os
#import magic
import urllib.request
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'C:/QMIND_19_20/flask_uploads'

app = Flask(__name__, template_folder='C:/QMIND_19_20/templates')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['mp4', 'png', 'jpg', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('test.html')

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			flash('File successfully uploaded')
			return redirect('/print')
		else:
			flash('Allowed file types are mp4, png, jpg, and jpeg')
			return redirect(request.url)

@app.route('/print')
def print_output():
	empty = True
	while(empty == True):
		if len(os.listdir("C:/QMIND_19_20/inputText/")) != 0:
			print("Not empty")
			empty=False
	with open("C:/QMIND_19_20/inputText/output.txt", "r") as textFile:
		inputString = textFile.readlines()
	inputString = ''.join(str(e) for e in inputString)
	flash(inputString)
	return redirect('/')

@app.route('/image', methods=['POST'])
def upload_image():
	print("we made it")
	return redirect('/print')

if __name__ == "__main__":
    app.run()