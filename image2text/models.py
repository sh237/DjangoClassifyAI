from django.db import models

import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from PIL import Image
import sys
import io, base64

graph = tf.compat.v1.get_default_graph()
class Photo(models.Model):
    image = models.ImageField(upload_to='photos')

    IMAGE_SIZE = 224
    MODEL_FILE_PATH = 'firstapp/ml_models/EfficientNetV2M_model.h5'
    
    def predict(self):
        model = None
        global graph
        with graph.as_default():
            model = load_model(self.MODEL_FILE_PATH)

            img_data = self.image.read()
            img_bin = io.BytesIO(img_data)

            image = Image.open(img_bin)
            image = image.convert("RGB")
            image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
            data = np.asarray(image) /255.0
            X = []
            X.append(data)
            X = np.array(X)
            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted] * 100)
            print(self.classes[predicted], percentage)
            return self.classes[predicted], percentage
