from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

from PIL import Image

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# View image
def img_show(img):
    pil_img = Image.fromarray((img))
    pil_img.show()

# If you want to view the image, uncomment the following.
# img = x_train[0]
# label = y_train[0]
# print(label)
#
# print(img.shape)
# img = img.reshape(28,28)
# print(img.shape)
#
# img_show(img)

# Flatten 28*28 images to a 784 vector for each image
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test  = x_test.reshape( x_test.shape[0] , num_pixels).astype('float32')

# Normalize inputs from 0-255 to 0-1
x_train = x_train/255
x_test  = x_test/255

# One hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test  = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Define baseline model
def baseline_model():
    model = Sequential()
    model.add(Dense( num_pixels, input_dim = num_pixels, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense( num_classes,                        kernel_initializer = 'normal', activation = 'softmax'))

    # compile model
    model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

# Build the model
model = baseline_model()

# Fit the model
model.fit( x_train, y_train, validation_data = (x_test, y_test), epochs = 10, batch_size = 200, verbose = 2)

scores = model.evaluate( x_test, y_test, verbose = 0 )
print( 'Baseline error: %.2f%%'%(100 - scores[1] * 100 ) )


