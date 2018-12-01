#import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import os 
#os.environ["CUDA_VISIBLE_DEVICES"]=""
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
X = pickle.load(open('x.pickle', 'rb'))
Y = pickle.load(open('y.pickle', 'rb'))

X = X/ 255.0

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model.fit(X,Y, batch_size=1, epochs=3, validation_split=0.1)
model.save('res.model')