import numpy as np
import tensorflow as tf
from keras.models import load_model
# from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def getPrediction(filename):
#  Load the best model
    model = load_model("models_dump\Final_Model.h5")
    path = filename
    img = tf.io.read_file(filename)

  # Decode the read file into a tensor & ensure 3 colour channels
  # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
    img = tf.image.decode_image(img, channels=3)

  # Resize the image (to the same size our model was trained on)
    img = tf.image.resize(img, size = [512, 512])

  # Rescale the image (get all values between 0 and 1)
    img = img/255.
    case = tf.expand_dims(img, axis=0)
    pred = model.predict(case)
    print("---------------Diagnosis is:", int(tf.round(pred[0][0])))
   
    return int(tf.round(pred[0][0]))
    

# pred = getPrediction('static/images/WBC-Malignant-Pre-001.jpg')



