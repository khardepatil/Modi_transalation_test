

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
import tensorflow as tf             
import numpy as np                          
from keras.preprocessing import image
 
img_width, img_height = 100,100

train_data_dir = 'Train/'
validation_data_dir = 'validation/'
nb_train_samples = 1000
nb_validation_samples = 100
epochs = 50
batch_size = 20

if K.image_data_format() == 'channel_first':
	input_shape = (3,img_width, img_height) 
else:
	input_shape = (img_width, img_height,3) 

train_datagen = ImageDataGenerator(
	rescale=1. /255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1. /255)



train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='binary')



#model = tf.keras.Sequential([
#                    tf.keras.layers.Flatten(input_shape=(100,100,1)),
#                    tf.keras.layers.Dense(activation='relu'),
#                    tf.keras.layers.Dense((28,28), activation='softmax')                        
#                ])



Model = Sequential()
Model.add(Conv2D(32, (3,3), input_shape=input_shape))
Model.add(Activation('relu')) 
Model.add(MaxPooling2D(pool_size=(2,2)))

Model.add(Conv2D(32, (3,3)))
Model.add(Activation('relu'))
Model.add(MaxPooling2D(pool_size=(2,2)))

Model.add(Conv2D(64, (3,3)))
Model.add(Activation('relu'))
Model.add(MaxPooling2D(pool_size=(2,2)))


Model.add(Flatten())
Model.add(Dense(64))  
Model.add(Activation('relu'))
Model.add(Dropout(0.5))
Model.add(Dense(1))
Model.add(Activation('sigmoid'))

Model.summary()

#Model.compile(loss='categorial_crossentropy' , optimizer='rmsprop',
#metrics=['accuracy'])
#Model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
#epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT,
#verbose=VERBOSE)
#score = model.evaluate(X_test,Y_test,
#batch_size=BATCH_SIZE, verbose=VERBOSE)
#print("Test Score::", score[0])
#print('test accuracy:',score[1])

Model.compile(loss='binary_crossentropy',
		optimizer='rmsprop',
		metrics=['accuracy'])

#Model.fit(
#   x = np.array(test_datagen),
#   y = np.array(validation_generator),
#   batch_size = 20,
#   epochs = 3)

Model.evaluate(
	train_generator,
	epochs=50,
	validation_data=validation_generator,
	validation_steps=nb_validation_samples // batch_size)

Model.save_weights('first_try.h5')

img_pred = image.load_img('bha_26.jpg', target_size = (100,100))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = Model.predict(img_pred)
print(rslt)	
print(img_pred)
