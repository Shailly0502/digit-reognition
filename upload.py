from flask import *  
import os
from flask import render_template
from flask import Flask, flash,request,redirect,url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing import image 
from keras.models import load_model
from keras.backend import set_session
import tensorflow as tf
import cv2 as cv


app = Flask(__name__)  
model=load_model('mnist.h5') 
UPLOAD_FOLDER=r"D:\naya wala folder\static\UPLOAD_FOLDER"

@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  
 
@app.route('/prediction', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']
        filename = secure_filename(f.filename)  
        f.save(os.path.join(UPLOAD_FOLDER, filename))
        test_image = image.load_img(UPLOAD_FOLDER+"/"+filename,target_size=(28,28))
        test_image = image.img_to_array(test_image)
        digit=image_test(test_image)
        return render_template('prediction.html', prediction_text='Number is : {} and Accuracy is {}'.format(np.argmax(digit),int(max(digit)*100)))


def image_test(img):
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img=img.reshape(1, 28, 28, 1)
            img = img/255.0
            digit = model.predict([img])[0]
            return (digit)
             


if __name__ == '__main__': 
    app.run(debug = True)