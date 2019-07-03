from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import pandas as pd
from sklearn.cluster import KMeans
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model, model_from_json

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')


config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

if C.network == 'resnet50':
	import keras_frcnn.resnet as nn



# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

ht = 224
wid = 224

def resize(img):
	
	num = 0
	#print(dirs)
	mean = 0

	im = img
	#print(item)
	w = np.size(im, 1)
	h = np.size(im, 0)
	row = h
	col = w
	if(max(h,w) <= ht):
	    delta_w = wid - w
	    delta_h = ht - h
	    ##

	    bot= im[row-2:row, 0:col]
	    #print(bot.shape)
	    mean= cv2.mean(bot)[0]
	    #print(cv2.mean(bot))
	    (left, top, right, bottom) = (0, 0, 0, delta_h-(delta_h//2))
	    dst = cv2.copyMakeBorder(im, top, bottom, left, right, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
	    ##
	    top1= im[0:2, 0:col]
	    mean= cv2.mean(top1)[0]
	    (left, top, right, bottom) = (0, delta_h//2, 0, 0)
	    dst = cv2.copyMakeBorder(dst, top, bottom, left, right, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
	     ##
	    left1= im[0:row, 0:2]
	    mean= cv2.mean(left1)[1]
	    (left, top, right, bottom) = (delta_w//2, 0, 0, 0)
	    dst = cv2.copyMakeBorder(dst, top, bottom, left, right, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
	    ##
	    right1 = im[0:row, col-2:col]
	    mean= cv2.mean(right1)[1]
	    (left, top, right, bottom) = (0, 0, delta_w-(delta_w//2), 0)
	    dst = cv2.copyMakeBorder(dst, top, bottom, left, right, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
	   
	elif(h > ht and w <= wid):
	    delta_w = wid - w
	     ##
	    left1= im[0:row, 0:2]
	    mean= cv2.mean(left1)[1]
	    (left, top, right, bottom) = (delta_w//2, 0, 0, 0)
	    dst = cv2.copyMakeBorder(im, top, bottom, left, right, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
	    ##
	    right1 = im[0:row, col-2:col]
	    mean= cv2.mean(right1)[1]
	    (left, top, right, bottom) = (0, 0, delta_w-(delta_w//2), 0)
	    dst = cv2.copyMakeBorder(dst, top, bottom, left, right, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )                            
	    #(left, top, right, bottom) = (delta_w//2, 0, delta_w-(delta_w//2), 0)
	    #dst = cv2.copyMakeBorder(im, top, bottom, left, right,  borderType= cv2.BORDER_REPLICATE )
	    dst = cv2.resize(dst, (wid, ht))
	    
	elif(h <=ht and w > wid):
	    delta_h = ht - h
	    bot= im[row-2:row, 0:col]
	    mean= cv2.mean(bot)[0]
	    (left, top, right, bottom) = (0, 0, 0, delta_h-(delta_h//2))
	    dst = cv2.copyMakeBorder(im, top, bottom, left, right, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
	    ##
	    top1= im[0:2, 0:col]
	    mean= cv2.mean(top1)[0]
	    (left, top, right, bottom) = (0, delta_h//2, 0, 0)
	    dst = cv2.copyMakeBorder(dst, top, bottom, left, right, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
	    #(left, top, right, bottom) = (0, delta_h//2, 0, delta_h-(delta_h//2))
	    #dst = cv2.copyMakeBorder(im, top, bottom, left, right,  borderType= cv2.BORDER_REPLICATE )
	    dst = cv2.resize(dst, (wid, ht))
	   
	else: 
	    if(h > w):
	        delta_w = h - w
	         ##
	        left1= im[0:row, 0:2]
	        mean= cv2.mean(left1)[1]
	        (left, top, right, bottom) = (delta_w//2, 0, 0, 0)
	        dst = cv2.copyMakeBorder(im, top, bottom, left, right, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
	        ##
	        right1 = im[0:row, col-2:col]
	        mean= cv2.mean(right1)[1]
	        (left, top, right, bottom) = (0, 0, delta_w-(delta_w//2), 0)
	        dst = cv2.copyMakeBorder(dst, top, bottom, left, right, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )  
	        #(left, top, right, bottom) = (delta_w//2, 0, delta_w-(delta_w//2), 0)
	        #dst = cv2.copyMakeBorder(im, top, bottom, left, right, borderType= cv2.BORDER_REPLICATE )
	        dst = cv2.resize(dst, (wid, ht))
	       
	    else: 
	        delta_h = w - h
	        bot= im[row-2:row, 0:col]
	        mean= cv2.mean(bot)[0]
	        (left, top, right, bottom) = (0, 0, 0, delta_h-(delta_h//2))
	        dst = cv2.copyMakeBorder(im, top, bottom, left, right, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
	        ##
	        top1= im[0:2, 0:col]
	        mean= cv2.mean(top1)[0]
	        (left, top, right, bottom) = (0, delta_h//2, 0, 0)
	        dst = cv2.copyMakeBorder(dst, top, bottom, left, right, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
	        #(left, top, right, bottom) = (0, delta_h//2, 0, delta_h-(delta_h//2))
	        #dst = cv2.copyMakeBorder(im, top, bottom, left, right,  borderType= cv2.BORDER_REPLICATE)
	        dst = cv2.resize(dst, (wid, ht))
	       
	num = num + 1
	#cv2.imwrite(os.path.join(dest_path , str(num) + '.jpg'), dst)
	#cv2.waitKey(0)    
	return dst

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

class_mapping = C.class_mapping

model_path = '../Classifier/class_model.h5'
model_weights_path = '../Classifier/class_weights.h5'
model_class = load_model(model_path)
time_tot = []


C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
	num_features = 1024

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

model_rpn = Model(img_input, rpn_layers)


print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.8

visualise = True
boxx = []

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue
	print(img_name)
	st = time.time()
	filepath = os.path.join(img_path,img_name)

	img = cv2.imread(filepath)
	X, ratio = format_img(img, C)

	if K.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))

	# get the feature maps and output from the RPN
	[Y1, Y2, F] = model_rpn.predict(X)

	R, probs = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)
	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]
	
	# apply the spatial pyramid pooling to the proposed regions
	bboxes = []
	probs1 = []
	
	var=0
	for i in range(R.shape[0]):
		if(probs[i]<bbox_threshold):
			continue
				
		(x, y, w, h) = R[i,:]

		bboxes.append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
		probs1.append(probs[i])
		var=var+1
		
	bbox = np.array(bboxes)
	bbox_prob = np.array(probs1)
	
	if(bbox != []):
		new_boxes = roi_helpers.non_max_suppression_fast3(bbox, bbox_prob, overlap_thresh=0.1, max_boxes = 100)[0]
	else:
		print('no boxes here')
		boxx.append((img_name, 'none','none','none','none' ))
		print('Elapsed time = {}'.format(time.time() - st))
		continue
	
	real_new_boxes = new_boxes
	
	for jk in range(new_boxes.shape[0]):
		
		(x1, y1, x2, y2) = new_boxes[jk,:]

		real_new_boxes[jk, :] = get_real_coordinates(ratio, x1, y1, x2, y2)

	cnt=0
	for index in range(real_new_boxes.shape[0]):

		real_x1, real_y1, real_x2, real_y2 = real_new_boxes[index, :]
		cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (0,0,255),2)

		x_1 = real_x1
		y_1 = real_y1
		x_2 = real_x2
		y_2 = real_y2
					
		crop_im = img[x_1:x_2 , y_1:y_2]
		
		resized_img = resize(crop_im)
		x = resized_img
		x = img_to_array(x)
		x = np.expand_dims(x, axis=0)
		array = model_class.predict(x)
		result = array[0]
		if(result > 0.5):
			answer = 1
		else:
			answer = 0
		if(answer == 0): 
			key = 'Knife'
		else:
			key = 'Scissor'
		real_x1, real_y1, real_x2, real_y2 = (x_1, y_1, x_2, y_2)
		cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (0,0,255),2)
		textLabel = '{}'.format(key)

		(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
		textOrg = (real_x1, real_y1-0)
		cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
	if not os.path.exists('./Final_Results'):
	    os.mkdir('./Final_Results')
	cv2.imwrite('./Final_Results/{}.png'.format(img_name),img)
	print('Elapsed time = {}'.format(time.time() - st))
	time_tot.append(time.time() - st)

df = pd.DataFrame(data = time_tot)
np.savetxt(r'./time.txt', df.values, fmt='%f')

