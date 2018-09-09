# Neural network tutorial using Keras

*Disclaimer: 제가 직접 작성한 글은 아니고, 해외 자료 다수를 응용하여 제가 이해한 바에 따라 재편집/번역 하였습니다. 제가 직접/간접적으로 이용한 모든 자료는 글 하단에 모두 링크하였습니다.*
<br />
*Disclaimer: The following is not my original writing. I have borrowed ideas from numerous sources both directly and indirectly. All the sources I used are linked at the bottom of this page. In particular, I have found [Machine Learning Mastery](https://machinelearningmastery.com/) the most helpful.*

<br />
데이터 사이언스의 'hello world' 격인 MNIST 데이터 세트를 이용하여 convolutional neural network와 LSTM neural network를 만들어 보겠습니다.

## MNIST handwritten digit recognition problem 
MNIST 손글씨 데이터셋은 Yann LeCun, Corinna Cortes 그리고 Christopher Burges에 의해 만들어 졌으며, machine learning을 사용하여 사람이 쓴 숫자가 무엇인지 구별해 내는 문제입니다. 각 이미지는 28 X 28의 픽셀로 되어 있으며, 데이터셋은 60,000개의 training 이미지와 또 다른 10,000개의 test 이미지로 구성되어 있습니다. 

## Download dataset 
데이터를 다운받을 수 있는 경로는 다양하며, 제가 사용할 방법은 Keras library에서 기본으로 제공하는 데이터셋을 다운 받는 것입니다. 

```python
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
데이터가 어떻게 생겼는지 한 번 보도록 하겠습니다. 

```python
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray((img))
    pil_img.show()

img = x_train[0]
label = y_train[0]
print(label)

print(img.shape)
img = img.reshape(28,28)
print(img.shape)

img_show(img)
```
휘갈겨 쓴 듯한 숫자 5가 보이실 것입니다. 사람은 숙련된 사고를 이용하여, 필기로 쓴 숫자를 쉽게 알아볼 수 있지만, 컴퓨터는 일련의 연습을 통해 숫자를 알아채도록 훈련해야 합니다. 지금부터 우리가 할 것이 바로 이 훈련 프로그램을 짜는 것입니다. 

## (Baseline) Multi layer perceptron model

Baseline이 될 [multi layer perceptron model](https://en.wikipedia.org/wiki/Multilayer_perceptron)을 만들어 보겠습니다. MLP는 말 그대로 multi layers (3개 이상의 layers) 갖고 있는 가장 기본적인 Neural network의 한 종류입니다. 저는 Keras library를 사용할 것이므로 필요한 Keras library를 `import`하도록 하겠습니다. 

```python
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
```

### Pre-processing
머신러닝에서 가장 중요한 단계인 pre-processing 단계에서는 input data가 보다 모델에 적합하도록 변형시키는 작업을 합니다. 우선 x 값 (input data) 	부터 정리해 보겠습니다. Training dataset의 이미지 한 장은 3-dimensional로 되어 있습니다 (instance, width, 그리고 height). MLP를 하기 위해서 가장 먼저 이미지 사이즈를 변형해야 합니다. 기존 28X28였던 이미지를 784픽셀이 되도록 바꾸어 보겠습니다. `numpy.reshape()`을 사용하면 이 작업을 수월하게 하실 수 있습니다. 

```python
num_pixels = x_train.shape[1] * x_train.shape[2]

x_train    = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test     = x_test.reshape( x_test.shape[0] , num_pixels).astype('float32')
```
한 개의 픽셀은 grey scale 정보를 표시하도록 되어 있어서, 0에서 255 사이의 숫자입니다. Neural network를 보다 심플하게 만들기 위해, 이 큰 숫자를 좀 더 comprehensible 한 수준으로 만드는 작업, 즉 normalize를 하도록 하겠습니다. 다양한 방법으로 normalize를 할 수 있지만, 저는 모든 숫자를 255로 나누도록 하겠습니다. 그 결과 모든 숫자들은 0 에서 1 사이의 숫자가 될 것입니다. 

```python
x_train = x_train/255
x_test  = x_test/255
``` 
준비가 거의 다 끝나갑니다. Input data를 조정했으니 이제는 output data를 조정할 차례입니다. 원래 output 값은 각 손글씨 숫자 이미지의 정답이 적혀 있으며 0 부터 9의 숫자가 적혀 있습니다. 이런 케이스는 classification 문제에서 multi-class classification 문제라고 불립니다. 다양한 카테고리 (0 부터 9라는 정답 카테고리) 중에 한 개(진짜 정답)를 골라야 하기 때문이죠. 문제를 좀 더 쉽게 만들기 위해 one-hot encoding이라는 기법을 사용하여 categorical value를 binary value로 바꾸어 주겠습니다. Keras에서 제공하는 `np_utils.to_categorical()` function이 저 대신 일을 하겠죠.
```python
y_train = np_utils.to_categorical(y_train)
y_test  = np_utils.to_categorical(y_test)

# Possible 정답의 카테고리 (종류) 개수를 저장합니다. 
num_classes = y_test.shape[1]
```

### Define & compile model
모든 준비가 끝났으니, 가볍게 모델을 만들어 보도록 하겠습니다. Keras는 TensorFlow를 기가막히게 쉽게 사용할 수 있도록 도와줍니다. 가볍게 한다고 약속했으니, 딱 1개의 hidden layer가 있는 모델을 만들어 보겠습니다. Keras를 사용하여 모델을 만드는 작업은 샌드위치 전문점인 Subway에서 주문을 하는 과정과 비슷하다고 생각하시면 됩니다. Subway에 가면 제일 먼저 주문을 넣죠.
<br />
* *나: 저 BLT 주문할꺼구요...흰 빵에 베이컨, 그 위에 양상치, 토마토, 마지막으로 이탈리안 소스 올려주세요*
* *점원: (주문한 데로 샌드위치 합체 중)... 나왔습니다.*
<br />

샌드위치 만들던 기억을 떠올리면 Keras의 주문구조가 단번에 이해될 것입니다. 
```python
def baseline_model():
    # 난 모델을 만들겠다, 어떤 모델? Sequential 모델 (==> Subway에서 맨 먼저 BLT를 주문하겠다고 얘기하는 과정 )
    model = Sequential()

    # 첫 번째 layer (1st hidden layer)를 더할거다. 어떤 layer? Dense
    model.add(Dense( num_pixels, input_dim = num_pixels, kernel_initializer = 'normal', activation = 'relu'))
    
    # 두 번째 layer는 정답을 부르기 위한 layer로 만들겠어. 그러므로 output shape이 num_classes이어야 하겠지
    model.add(Dense( num_classes,                        kernel_initializer = 'normal', activation = 'softmax'))

    # 더할 layer가 더 없으면
    # 모델 합체!
    model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model
```
`Sequential()`모델은 Keras에서 사용가능 한 가장 심플한 모델로, 그냥 내가 만들고 싶은 layers를 쭉 나열하면 합체가 되는 모델입니다. 다양한 종류와 역할을 하는 각기 다른 layers를 나열하는 방법이나 순서가 각각 다른 모델의 특징이 됩니다. 그런데 그 어떤 모델이던 가장 첫번째 layer가 해야 하는 역할이 있습니다. 바로 input data의 shape을 모델에게 말해주는 것입니다. 
<br />
<br />
비슷한 예시로 Subway에서 어떤 주문을 하던 가장 먼저 말해야 하는것은 빵 종류입니다. 빵을 고르지 않고, '양파랑 피망 빼주세요' 라고 말을 하면, 점원이 '빵은 무얼로 드릴까요' 라며 빵부터 다시 순서를 잡아줄 것입니다. 빵 없이는 샌드위치 주문이 시작될 수가 없는 거죠. 
<br />
<br />
Input data의 shape를 알려주는 방법은 여러가지가 (`input_dim` 아니면 `input_shape`) 있을 수 있습니다. 저는 `input_dim`이라는 argument를 사용했습니다. 이 `input_dim`에는 integer를 넣으셔야 합니다. 
#### Layers의 종류들: Dense, Dropout
Dense는 기본 layer라고 할 수 있습니다. 이름이 왜 dense이냐면, 이 layer는 fully connected 되었기 때문입니다. Fully connected 되었다는 의미는, 이 전(previous) layer에서 온 모든 데이터 값들이 하나도 빠짐 없이 고대로 이 layer에 도착한다는 의미입니다. 억지스럽게라도 Subway 예시와 연결해 보자면, 빵 면적 위에 마요네즈를 모두 펴 바른 케이스가 되겠네요. 모든 면적에 마요네즈를 바르고 똑같은 크기의 빵으로 그 위를 덮는다면, 위에 올라간 빵 모든 면적에 그대로 마요네즈가 묻겠죠. 간혹 특이한 취향의 손님은 마요네즈를 중앙에만 발라달라던지, 아니면 왼쪽 코너 아래에만 발라달라는 주문을 할 수도 있을텐데, 이런 케이스는 fully connected라고 할 수 없습니다. 아무튼 fully connected라서 dense라고 불립니다. 
<br />
<br />
Dropout은 일부러 데이터 몇 개를 버려버리는 layer입니다. 아무리 고민을 해보아도 Subway 예시와 연결 지을 고리를 찾을 수가 없어 그냥 다른 적절한 예시를 들...려고 했는데, 예시가 필요하지 않을 정도로 간단한 컨셉이라 그냥 넘어가겠습니다. Overfitting 문제를 해결하기 위하여 Dropout layer를 사용합니다. 
<br />
<br />
모델 구성을 보시면, `kernel_initializer` 라던지 `activation` argument가 보이는데, 이 부분은 다른 기회에 좀 더 자세하게 다루도록 하고, 지금은 일단 위와 같이 setting 했다고만 알고 넘어가도록 합시다. 

#### Compile (합체)
Compile 단계에서는 세 가지 중요한 attribute를 pass해야 합니다. 
* Model optimizer(`optimizer`): Search technique으로, 모델의 parameter를 어떻게 업데이트 할 지를 정합니다.  
* Loss function(`loss`): Weight를 조정하며, 모델을 evaluate 합니다.
* Metrics(`metrics`): 무얼 측정 할 것인지 정합니다 (예: accuracy)

### Build model
위에 쓴 `baseline_model`function을 사용하여, 모델을 build 하도록 하겠습니다. 

```python
model = baseline_model()
model.fit( x_train, y_train, validation_data = (x_test, y_test), epochs = 10, batch_size = 200, verbose = 2)

# 모델 스코어 점검
scores = model.evaluate( x_test, y_test, verbose = 0 )
print( 'Baseline error: %.2f%%'%(100 - scores[1] * 100 ) )
```
이제 떨리는 마음으로 모델을 실행시켜 봅니다. 
```python
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
 - 4s - loss: 0.2821 - acc: 0.9202 - val_loss: 0.1378 - val_acc: 0.9606
Epoch 2/10
 - 3s - loss: 0.1106 - acc: 0.9679 - val_loss: 0.0939 - val_acc: 0.9709
Epoch 3/10
 - 3s - loss: 0.0717 - acc: 0.9794 - val_loss: 0.0743 - val_acc: 0.9781
Epoch 4/10
 - 3s - loss: 0.0490 - acc: 0.9865 - val_loss: 0.0722 - val_acc: 0.9777
Epoch 5/10
 - 3s - loss: 0.0367 - acc: 0.9895 - val_loss: 0.0621 - val_acc: 0.9804
Epoch 6/10
 - 3s - loss: 0.0258 - acc: 0.9932 - val_loss: 0.0619 - val_acc: 0.9798
Epoch 7/10
 - 3s - loss: 0.0181 - acc: 0.9957 - val_loss: 0.0621 - val_acc: 0.9796
Epoch 8/10
 - 3s - loss: 0.0140 - acc: 0.9968 - val_loss: 0.0577 - val_acc: 0.9818
Epoch 9/10
 - 3s - loss: 0.0107 - acc: 0.9977 - val_loss: 0.0634 - val_acc: 0.9813
Epoch 10/10
 - 3s - loss: 0.0080 - acc: 0.9984 - val_loss: 0.0591 - val_acc: 0.9821
Baseline error: 1.79%
```
Error가 1.79%로 나온 것을 확인할 수 있습니다. 지금까지 설명드린 내용은 `mnist_mlp.py` 파일에서 full-version을 보실 수 있습니다. 
<br />
<br />
다음 단계에서는 좀 더 complex한 모델을 사용하여, 이 accuracy를 어디까지 높일 수 있을지 보도록 하겠습니다.

## Convolutional neural network model (Light version)
[Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)은 조금 더 advanced된 neural network로 이미지 서칭등에 자주 사용되는 아주 좋은 모델입니다. 일부 준비 과정은 위에 언급한 baselinea 모델과 동일하므로, repetitive한 순서는 설명을 생략하겠습니다.
<br />
<br />
Baseline 모델을 했을 때와 마찬가지로 pre-processing 과정이 필요한데, 이는 CNN 모델에 사용하기 적합한 데이터 형태로 만들어야 하기 때문입니다. 이번에는 일부 (input data)를 다르게 변형시켜보겠습니다. 
```python
# 데이터를 로딩하고
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Input 데이터 형태를 바꿔줍니다.
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test  = x_test.reshape( x_test.shape[0], 1, 28, 28). astype('float32')

# 그리고 255로 나누어 normalize를 합니다. 
x_train = x_train / 255
x_test  = x_test  / 255

# Y 값을 바꿔줍니다.
y_train = np_utils.to_categorical(y_train)
y_test  = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
```
### CNN모델의 기본 순서
아주 간략한 버전으로 CNN모델에 어떤 Layer가 필요하고, 어떤 순서로 쌓이는지 설명드리도록 하겠습니다. 
1. Input layer: 모든 샌드위치 주문은 빵 주문부터 시작한다는 것을 기억하시는지요. CNN에서도 가장 쳣 번째 layer는 input shape를 지정합니다. Baseline모델에서는 `Dense`라는 layer를 사용해지만 CNN에서는 Convolution2D(`Conv2D`)라는 종류를 씁니다. 
2. Pooling layer: 두 번째 layer로 MaxPooling2D(`MaxPooling2D`)라는 종류를 씁니다. 
3. Regularization layer: 그 위에 올라갈 layera는 regularization layer로 dropout(`Dropout`)라는 종류를 씁니다.
4. Flatten layer: 2D matrix를 vector로 만들어줄 layer입니다.
5. Connected layer: 앞서 있었던 layer값들이 모두 제대로 넘어 올 수 있도록 도와주는 역할을 합니다. 
6. Output layer: Baseline과 마찬가지로 마지막 layer에서는 모델의 측정값이 나와야 합니다. 여기에서는 shape를 10으로 조정, 컴퓨터로 하여금 손글씨 0 부터 9 중 정답을 고르라고 시킬 것입니다. 

```python
def cnn_simple_model():
    # create model
    model = Sequential()
    model.add( Conv2D(32, (5, 5), input_shape = (1, 28, 28), activation = 'relu' )  )
    model.add( MaxPooling2D(pool_size = (2,2) )  )
    model.add( Dropout(0.2) )
    model.add( Flatten() )
    model.add( Dense(128, activation = 'relu')  )
    model.add( Dense(num_classes, activation = 'softmax'  ))

    # compile model
    model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']  )
    return model

# build the model
model = cnn_simple_model()

# fit the model
model.fit( x_train, y_train, validation_data = (x_test, y_test) , epochs = 10, batch_size = 200, verbose = 2 )

# final eval of the model
scores = model.evaluate( x_test, y_test, verbose = 0 )

print( 'simple CNN error: %.2f%%'% ( 100 - scores[1] * 100 ) )
```
자 이제 두근두근하는 마음으로 모델을 돌려봅시다. 
```python
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
 - 123s - loss: 0.2534 - acc: 0.9266 - val_loss: 0.0777 - val_acc: 0.9767
Epoch 2/10
 - 115s - loss: 0.0734 - acc: 0.9779 - val_loss: 0.0541 - val_acc: 0.9831
Epoch 3/10
 - 117s - loss: 0.0510 - acc: 0.9843 - val_loss: 0.0442 - val_acc: 0.9863
Epoch 4/10
 - 116s - loss: 0.0404 - acc: 0.9876 - val_loss: 0.0360 - val_acc: 0.9879
Epoch 5/10
 - 126s - loss: 0.0330 - acc: 0.9895 - val_loss: 0.0408 - val_acc: 0.9863
Epoch 6/10
 - 123s - loss: 0.0278 - acc: 0.9910 - val_loss: 0.0295 - val_acc: 0.9903
Epoch 7/10
 - 123s - loss: 0.0221 - acc: 0.9930 - val_loss: 0.0291 - val_acc: 0.9906
Epoch 8/10
 - 121s - loss: 0.0189 - acc: 0.9939 - val_loss: 0.0280 - val_acc: 0.9904
Epoch 9/10
 - 124s - loss: 0.0154 - acc: 0.9951 - val_loss: 0.0294 - val_acc: 0.9903
Epoch 10/10
 - 117s - loss: 0.0135 - acc: 0.9958 - val_loss: 0.0296 - val_acc: 0.9900
simple CNN error: 1.00%
```
Error가 1.00%로 나온 것을 확인할 수 있습니다. Baseline 모델보다 약 0.8%p나 떨어진 숫자입니다. 지금까지 설명드린 내용은 `covnn_mnist_light.py` 파일에서 full-version을 보실 수 있습니다. 

## Convolutional neural network model (Full version)
[Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)은 조금 더 advanced된 neural network로 이미지 서칭등에 자주 사용되는 아주 좋은 모델입니다. 일부 준비 과정은 위에 언급한 baselinea 모델과 동일하므로, repetitive한 순서는 설명을 생략하겠습니다. 위에서 연습했던 light 버전 위에 몇개의 layer를 더 올려보고 accuracy에 차이가 있는지 보도록 하겠습니다. 

```python
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

```
이제 조금 더 시간이 걸릴 것입니다. 잠깐 자리에서 일어나서 커피 한잔 타서 오셔도 될듯 합니다. 
```python
Train on 60000 samples, validate on 10000 samples
Epoch 1/10

  200/60000 [..............................] - ETA: 3:29 - loss: 2.2929 - acc: 0.1400
  400/60000 [..............................] - ETA: 2:35 - loss: 2.2748 - acc: 0.1500
  600/60000 [..............................] - ETA: 2:17 - loss: 2.2483 - acc: 0.1833
  800/60000 [..............................] - ETA: 2:08 - loss: 2.2242 - acc: 0.2087
 1000/60000 [..............................] - ETA: 2:03 - loss: 2.1991 - acc: 0.2300
 ...
 58000/60000 [============================>.] - ETA: 3s - loss: 0.0154 - acc: 0.9947
58200/60000 [============================>.] - ETA: 3s - loss: 0.0154 - acc: 0.9947
58400/60000 [============================>.] - ETA: 2s - loss: 0.0155 - acc: 0.9946
58600/60000 [============================>.] - ETA: 2s - loss: 0.0155 - acc: 0.9946
58800/60000 [============================>.] - ETA: 2s - loss: 0.0155 - acc: 0.9946
59000/60000 [============================>.] - ETA: 1s - loss: 0.0154 - acc: 0.9946
59200/60000 [============================>.] - ETA: 1s - loss: 0.0155 - acc: 0.9946
59400/60000 [============================>.] - ETA: 1s - loss: 0.0155 - acc: 0.9946
59600/60000 [============================>.] - ETA: 0s - loss: 0.0154 - acc: 0.9946
59800/60000 [============================>.] - ETA: 0s - loss: 0.0155 - acc: 0.9946
60000/60000 [==============================] - 116s 2ms/step - loss: 0.0155 - acc: 0.9946 - val_loss: 0.0292 - val_acc: 0.9910
Larger CNN error: 0.90%
```
Light version보다 약 0.1%p나 나아진 것을 보실 수 있습니다. 이렇듯 layer를 어떻게 어떤 모양으로 더하냐(`add`)에 따라 모델의 performance가 달라질 수 있습니다. 다음 예시는 LSTM 모델입니다. 결과부터 말하자면, 아직 optimize가 되지 않아, CNN_full_version보다 accuracy가 낮습니다. 여기서 읽는 것을 멈추셔도 되고, 뭐가 어떻게 나쁜지 구경하는 마음으로 조금 더 읽어주셔도 좋습니다. 

## LSTM model
LSTM 모델의 결과부터 말하자면, 아직 optimize가 되지 않아, CNN_full_version보다 accuracy가 낮습니다. 여기서 읽는 것을 멈추셔도 되고, 뭐가 어떻게 나쁜지 구경하는 마음으로 조금 더 읽어주셔도 좋습니다.
<br />
<br />
우선 필요한 keras library를 모두 다운받습니다. 

```python
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
from keras.utils import np_uti
```
<br />
<br />
위에 CNN을 하면서 진행헀던 pre-processing을 거치는데, 데이터 형태를 LSTM 모델에 맞게 일부 변형해보겠습니다. 일단 데이터를 받습니다. 

```python
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
```

모델을 define하고 compile 할 차례입니다.

```python
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
```

제가 현재까지 받은 최저의 error rate는 3.58%입니다. 


### Source
* https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
* https://machinelearningmastery.com/build-multi-layer-perceptron-neural-network-models-keras/
* https://machinelearningmastery.com/recurrent-neural-network-algorithms-for-deep-learning/
* https://github.com/keras-team/keras/issues/2645
* https://stackoverflow.com/questions/46305252/valueerror-dimension-1-must-be-in-the-range-0-2-in-keras
* https://stackoverflow.com/questions/44410135/keras-conv2d-own-filters
* https://www.saama.com/blog/different-kinds-convolutional-filters/
* https://stackoverflow.com/questions/48243360/how-to-determine-the-filter-parameter-in-the-keras-conv2d-function
* https://github.com/ar-ms/lstm-mnist/blob/master/lstm_classifier.py
* https://github.com/ar-ms/lstm-mnist