# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:28:42 2020

@author: Colin Cumming
"""

from flask import Flask

UPLOAD_FOLDER = 'C:\\QMIND_19_20\\flask_input'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024