from Classifier_initialization import *
from keras.callbacks import ModelCheckpoint

epochs = 300

new_model = Sequential()

model = applications.VGG16(include_top=False, weights=None, input_shape = (224,224,3))
model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
print('Loaded Weights successfully')



new_model.add(model)
# build a classifier model to put on top of the convolutional model

new_model.add(Flatten(input_shape=model.output_shape[1:]))
new_model.add(Dense(256, activation='relu', name = 'fc_1'))
new_model.add(Dropout(0.5))
new_model.add(Dense(1, activation='sigmoid', name = 'fc_2'))

new_model.load_weights(top_model_weights_path, by_name=True)
print('Loaded weights of tt_result')


for layer in new_model.layers[:-4]:
    layer.trainable = False

for layer in new_model.layers:
    print(layer, layer.trainable)

print(new_model.summary())

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1. / 255)
if not os.path.exists('./models_bottom'):
    os.mkdir('./models_bottom')
checkpoint = ModelCheckpoint('./models_bottom/model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  
new_model.compile(loss='binary_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
new_model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples, callbacks = [checkpoint])

new_model.save_weights("class_weights.h5") 
new_model.save("class_model.h5")   
print('done with complete classification training')


