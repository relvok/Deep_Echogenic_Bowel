
# coding: utf-8

# In[ ]:


import cv2

import tensorflow as tf
import numpy as np
import os
from math import *
import random
import re
from PIL import Image
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
from keras.models import load_model
from keras.preprocessing import image

import numpy as np
from pylab import *
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *
#from tensorflow.keras.engine import Layer
from tensorflow.keras.applications.vgg16 import *
from tensorflow.keras.models import *
#from tensorflow.keras.applications.imagenet_utils import _obtain_input_shape
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Cropping2D, Conv2D
from tensorflow.keras.layers import Input, Add, Dropout, Permute, add
from tensorflow.compat.v1.layers import conv2d_transpose
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

tf.compat.v1.enable_eager_execution()

class Abd_Mask():
    def get_small_unet(self,n_filters = 16, bn = True, dilation_rate = 1):
        '''Validation Image data generator
            Inputs: 
                n_filters - base convolution filters
                bn - flag to set batch normalization
                dilation_rate - convolution dilation rate
            Output: Unet keras Model
        '''
        #Define input batch shape
        batch_shape=(256,256,3)
        inputs = Input(batch_shape=(2, 256,256, 3))
        print(inputs)

        conv1 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(inputs)
        if bn:
            conv1 = BatchNormalization()(conv1)

        conv1 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv1)
        if bn:
            conv1 = BatchNormalization()(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv1)

        conv2 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool1)
        if bn:
            conv2 = BatchNormalization()(conv2)

        conv2 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv2)
        if bn:
            conv2 = BatchNormalization()(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv2)

        conv3 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool2)
        if bn:
            conv3 = BatchNormalization()(conv3)

        conv3 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv3)
        if bn:
            conv3 = BatchNormalization()(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv3)

        conv4 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool3)
        if bn:
            conv4 = BatchNormalization()(conv4)

        conv4 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv4)
        if bn:
            conv4 = BatchNormalization()(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv4)

        conv5 = Conv2D(n_filters * 16, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool4)
        if bn:
            conv5 = BatchNormalization()(conv5)

        conv5 = Conv2D(n_filters * 16, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv5)
        if bn:
            conv5 = BatchNormalization()(conv5)

        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)

        conv6 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up6)
        if bn:
            conv6 = BatchNormalization()(conv6)

        conv6 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv6)
        if bn:
            conv6 = BatchNormalization()(conv6)

        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)

        conv7 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up7)
        if bn:
            conv7 = BatchNormalization()(conv7)

        conv7 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv7)
        if bn:
            conv7 = BatchNormalization()(conv7)

        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)

        conv8 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up8)
        if bn:
            conv8 = BatchNormalization()(conv8)

        conv8 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv8)
        if bn:
            conv8 = BatchNormalization()(conv8)

        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)

        conv9 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up9)
        if bn:
            conv9 = BatchNormalization()(conv9)

        conv9 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv9)
        if bn:
            conv9 = BatchNormalization()(conv9)

        conv10 = Conv2D(32, (1, 1), activation='softmax', padding = 'same', dilation_rate = dilation_rate)(conv9)

        model = Model(inputs=inputs, outputs=conv10)
        if bn:
            conv10 = BatchNormalization()(conv10)

        conv11 = Conv2D(3, (1, 1), activation='softmax', padding = 'same', dilation_rate = dilation_rate)(conv10)

        model = Model(inputs=inputs, outputs=conv11)

        return model



    # Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
    # -> the score is computed for each class separately and then summed
    # alpha=beta=0.5 : dice coefficient
    # alpha=beta=1   : tanimoto coefficient (also known as jaccard)
    # alpha+beta=1   : produces set of F*-scores
    # implemented by E. Moebel, 06/04/18
    def tversky_loss(self,y_true, y_pred):
        alpha = 0.5
        beta  = 0.5

        ones = K.ones(K.shape(y_true))
        p0 = y_pred      # proba that voxels are class i
        p1 = ones-y_pred # proba that voxels are not class i
        g0 = y_true
        g1 = ones-y_true

        num = K.sum(p0*g0, (0,1,2,3))
        den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))

        T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]

        Ncl = K.cast(K.shape(y_true)[-1], 'float32')
        return Ncl-T

    def dice_coef(self,y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


    def dice_coef_loss(self,y_true, y_pred):
        return 1.-self.dice_coef(y_true, y_pred)



    def load_image(self,img_path, show=True):

        img = image.load_img(img_path, target_size=(256, 256))
        img_tensor = image.img_to_array(img)                    # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.                                      # imshow expects values in the range [0, 1]



        return img_tensor
    def predict_image(self,img_path,weight_path):

        model = self.get_small_unet(n_filters = 32)
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[self.tversky_loss,self.dice_coef,'accuracy'])
        model.load_weights(weight_path)
        new_image = self.load_image(img_path)

        pred = model.predict(new_image)

        opencvImage = np.array(image.array_to_img(pred[0] < 0.14))
        cv2.imshow('opencv',opencvImage)

        return opencvImage



    def get_abdomen_mask(self,filename, frame, weights_path, height,width):
        x_mask = []; y_mask =[]
        cv2.imwrite(filename, frame)  
        mask = self.predict_image(filename, weights_path)
        #tf.compat.v1.disable_eager_execution()
        r_query = 0
        g_query = 0
        b_query = 0
        x_mask, y_mask = (np.where((mask[:,:,0] == r_query) & (mask[:,:,1] == g_query)
                                             & (mask[:,:,2] == b_query)))

        y_mask = np.floor(y_mask).astype(int)
        x_mask = np.floor(x_mask).astype(int)
        frame = cv2.resize(frame, (256,256),interpolation = cv2.INTER_AREA)
        img = np.zeros((256,256,3))
        img[x_mask, y_mask] = frame[x_mask,y_mask]

        return x_mask,y_mask, img

