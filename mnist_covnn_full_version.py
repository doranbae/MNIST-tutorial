import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test  = x_test.reshape( x_test.shape[0], 1, 28, 28). astype('float32')

# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test  = x_test  / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test  = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

# next define the model

def cnn_larger_model():
    # create_model
    model = Sequential()
    model.add( Conv2D(30, (5, 5), input_shape = (1, 28, 28), activation = 'relu' ) )
    model.add( MaxPooling2D(pool_size = (2, 2))  )
    model.add( Conv2D(15, (3, 3), activation = 'relu') )
    model.add( Dropout(0.2) )
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(num_classes, activation = 'softmax'))

    # compile model
    model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

# build the model
model = cnn_larger_model()

# fit the model
model.fit( x_train, y_train, validation_data = (x_test, y_test) , epochs = 10, batch_size = 200, verbose = 2 )

# final eval of the model
scores = model.evaluate( x_test, y_test, verbose = 0 )

print( 'Full CNN error: %.2f%%'% ( 100 - scores[1] * 100 ) )