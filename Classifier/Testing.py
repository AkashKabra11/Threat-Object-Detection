import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model, model_from_json
import time

start = time.time()

#Define Path
model_path = 'class_model.h5'
model_weights_path = 'class_weights.h5'
test_path = 'data/test'

vgg_model = applications.VGG16(include_top=False, weights=None, input_shape = (224,224,3))
vgg_model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

model = load_model(model_path)        
model.load_weights(model_weights_path)

vgg_model = 
#Define image parameters
img_width, img_height = 224, 224

#This function will forward propogate our trained model and will predict knife or scissor depending on whether result is above or below 0.5
#Note that this is a binary class prediction model and will work for two classes only 
def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  y = vgg_model.predict(x)
  array = model.predict(y)
  result = array[0]
  if(result > 0.5):
     answer = 1
  else:
     answer = 0
  if(answer == 0): 
     print('got Knife')
  else:
     print('got Scissor')


file_list = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
file_list.sort()

for fname in file_list:
    print('Result for file {} is:'.format(os.path.join(test_path, fname)))
    result = predict(os.path.join(test_path, fname)); 
    print(" ")


end = time.time()
dur = end-start

if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")
