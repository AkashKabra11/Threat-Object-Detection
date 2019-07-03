from Classifier_initialization import *
from keras.callbacks import ModelCheckpoint
epochs = 300 


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    print('Adding VGG layer')
    model = applications.VGG16(include_top=False, weights=None, input_shape = (224, 224, 3))
    model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    print('Loaded Weights successfully')
	
	#Take the path of training directory & generates batches of augmented data
    generator = datagen.flow_from_directory( train_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode=None, shuffle=False)
        
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size)
    
	#Take the path of training directory & generates batches of augmented data
    generator = datagen.flow_from_directory( validation_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode=None, shuffle=False)
        
    bottleneck_features_validation = model.predict_generator( generator, nb_validation_samples // batch_size)
    
    return bottleneck_features_train, bottleneck_features_validation

def train_top_model(bottleneck_features_train, bottleneck_features_validation):
    train_data = bottleneck_features_train
    train_labels = np.array( [0] * (nb_train_samples_knife) + [1] * (nb_train_samples_scis))

    validation_data = bottleneck_features_validation
    validation_labels = np.array( [0] * (nb_validation_samples_knife) + [1] * (nb_validation_samples_scis))

    model = Sequential()
    vgg_model = applications.VGG16(include_top=False, weights=None, input_shape = (224,224,3))
    vgg_model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
	print('Loaded Weights successfully')
	model.add(vgg__model)
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu', name = 'fc_1'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', name = 'fc_2'))
    
    new_model.load_weights(top_model_weights_path, by_name=True)
		print(new_model.summary())
	
	
    for layer in new_model.layers[:-4]:
        layer.trainable = False

    for layer in new_model.layers:
        print(layer, layer.trainable)
	
    if not os.path.exists('./models'):
    	os.mkdir('./models')
    checkpoint = ModelCheckpoint('./models/model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(validation_data, validation_labels), callbacks = [checkpoint])
    model.save_weights(top_model_weights_path)

t1, t2 = save_bottlebeck_features()
train_top_model(t1, t2)
print('Done with training the top layers of the classifier')

