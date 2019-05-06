import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


# Intersection over Union for Objects
def IoU(y_true, y_pred, tresh=1e-10):
    Intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    Union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - Intersection
    return K.mean( (Intersection + tresh) / (Union + tresh), axis=0)
# Intersection over Union for Background
def back_IoU(y_true, y_pred):
    return IoU(1-y_true, 1-y_pred)
# Loss function
def IoU_loss(in_gt, in_pred):
    #return 2 - back_IoU(in_gt, in_pred) - IoU(in_gt, in_pred)
    return 1 - IoU(in_gt, in_pred)
 

def unet(pretrained_weights = None,input_size = (768,768,1)):
    inputs = Input(input_size)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)

    u5 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same') (u5)
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same') (c5)

    u6 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(16, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(16, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c1], axis=3)
    c7 = Conv2D(8, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(8, (3, 3), activation='relu', padding='same') (c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss= IoU_loss, metrics=[IoU, back_IoU])
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
