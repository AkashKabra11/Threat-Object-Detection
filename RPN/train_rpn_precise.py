from __future__ import division
import os
import random
import pprint
import keras
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import numpy
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import data_generators
from keras_frcnn import config as config
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils


# gpu setting
if 'tensorflow' == K.backend():
    import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config2 = tf.ConfigProto()
config2.gpu_options.allow_growth = True
set_session(tf.Session(config=config2))

# option parsar
parser = OptionParser()
parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
				default="pascal_voc")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=300)
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn_rpn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to store all the metadata related to the training (to be used when testing).",
				default="config.pickle")

(options, args) = parser.parse_args()
    
# we will train from pascal voc 2007
# you have to pass the directory of VOC with -p
if not options.train_path:   # if filename is not given
	parser.error('Error: path to training data must be specified. Pass --path to command line')


if options.parser == 'simple':
	from keras_frcnn.simple_parser import get_data
else:
	raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()

# set data augmentation
C.model_path = options.output_weight_path
C.num_rois = int(options.num_rois)

# we will use resnet. may change to vgg
if options.network == 'resnet50':
	from keras_frcnn import resnet as nn
	C.network = 'resnet50'
else:
	print('Not a valid model')
	raise ValueError

all_imgs, classes_count, _ = get_data(options.train_path)

# add background class as 21st class
if 'bg' not in classes_count:
	classes_count['bg'] = 0

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_imgs)
num_imgs = len(all_imgs)

# split to train and val
train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,K.image_dim_ordering(), mode='val')

# set input shape
input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# create rpn model here
# define the base network (resnet here)

shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
# rpn outputs regression and cls
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)
model_rpn = Model(img_input, rpn[:2])

#load weights from pretrain
try:
	print('loading weights from {}'.format(C.base_net_weights))
	model_rpn.load_weights(C.base_net_weights, by_name=True)
	print("loaded weights!")
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

# compile model
optimizer = Adam(lr=1e-5, clipnorm=0.001)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])

# write training misc here
num_epochs = int(options.num_epochs)
iter_num = 0
epoch_length =1000 

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

#best_loss = np.Inf

print('Starting training')
#vis = True
# Create target Directory if don't exist
if not os.path.exists('./models/rpn'):
    os.mkdir('./models')
    os.mkdir('./models/rpn')
Callbacks=keras.callbacks.ModelCheckpoint("./models/rpn/rpn."+options.network+".weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=4)
callback=[Callbacks]
history = model_rpn.fit_generator(data_gen_train,
                    epochs=num_epochs, validation_data=data_gen_val,
                    steps_per_epoch=epoch_length),callbacks=callback, validation_steps=10)
model_rpn.save_weights(C.model_path)
loss_history = history.history["val_loss"]

numpy_loss_history = numpy.array(epoch + " " + loss_history)
numpy.savetxt(options.network+"_rpn_loss_history.txt", numpy_loss_history, delimiter=",")
