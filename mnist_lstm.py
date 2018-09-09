import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
from keras.utils import np_uti

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.array(x_train).reshape((-1, 28, 28))
x_test  = np.array(x_test).reshape((-1, 28, 28))

# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test  = x_test  / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test  = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

def lstm_model():
    model = Sequential()
    model.add( LSTM(512, input_shape = (28, 28) ) )
    model.add( Dense(num_classes, activation = 'softmax') )

    # compile
    model.compile( loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'] )

    return model

model = lstm_model()
model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 200, epochs = 10, verbose = 2 )

# final eval of the model
scores = model.evaluate( x_test, y_test, verbose = 0 )

print( 'LSTM model error: %.2f%%'% ( 100 - scores[1] * 100 ) )

