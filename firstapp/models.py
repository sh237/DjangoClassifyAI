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
    classes = ['Nike_Air_Huarache_Light_306127-480','Nike_Air_Force_1_DD8959-100',
                'Nike_Air_Max_1_908375-100', 'Nike_Air_Max_95_Essential_CI3705-001',
                'Nike_Air_Max_90_CD0881-103','Nike_Air_Max_97_921826-001',
                'Nike_Air_Jordan_136066-041','Nike_Cortez_Basic_Leather_819719',
                'Nike_React_Element_55_BQ6166-101','Nike_Air_Huarache_DD1068-002',
                'Nike_Challenger_Og_CW7645','Nike_Blazer_Mid_77_BQ6806',
                'Nike_Dbreak_DB4635','Nike_Sb_Dunk_DD1391-100',
                'Nike_Air_Monarch_4_415445-002','Nike_Court_Royale_Sl_844802',
                'Nike_Air_Presto_CT3550','Nike_Venture_Runner_CQ4557',
                'Nike_Tanjun_DJ6258']  #クラスの設定
    num_classes = len(classes) #クラス数
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
