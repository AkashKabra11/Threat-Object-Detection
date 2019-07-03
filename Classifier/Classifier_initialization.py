import numpy as np
import os
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import optimizers
from keras.models import Model
from keras.backend import manual_variable_initialization
from keras.preprocessing.image import load_img, img_to_array

# dimensions of our image
img_width, img_height = 224 , 224
top_model_weights_path = 'tt_result.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples_knife = 1000
nb_train_samples_scis = 1000
nb_validation_samples_knife = 399
nb_validation_samples_scis = 279
nb_train_samples = nb_train_samples_knife + nb_train_samples_scis
nb_validation_samples = nb_validation_samples_knife + nb_validation_samples_scis
batch_size = 16



