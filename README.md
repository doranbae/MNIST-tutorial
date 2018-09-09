# Neural network tutorial using Keras

* Disclaimer: 제가 직접 작성한 글은 아니고, 해외 자료 다 수를 응용하여 제가 이해한 바에 따라 재편집/번역 하였습니다. 제가 직접/간접 이용한 모든 자료는 글 하단에 모두 링크하였습니다. *
* Disclaimer: The following is not my original writing. I have borrowed ideas from numerous sources online both directly and indirectly. All the sources I used are linked at the bottom of this page. In particular, I have found [Machine Learning Mastery](https://machinelearningmastery.com/) the most helpful. *

<br />
데이터 사이언스의 'hello world' 격인 MNIST 데이터 세트를 이용하여 convolutional neural network와 LSTM neural network를 만들어 보겠습니다.

## MNIST handwritten digit recognition problem 
MNIST 손글씨 데이터셋은 Yann LeCun, Corinna Cortes 그리고 Christopher Burges에 의해 만들어 졌으며, machine learning을 사용하여 사람이 쓴 숫자가 무엇인지 구별해 내는 문제입니다. 각 이미지는 28 X 28의 픽셀로 되어 있으며, 데이터셋은 60,000개의 training 이미지와 또 다른 10,000개의 test 이미지로 구성되어 있습니다. 

## Download dataset 
데이터를 다운받을 수 있는 경로는 다양하며, 제가 사용할 방법은 Keras library에서 기본으로 제공하는 데이터셋을 다운 받는 것입니다. 

```python
s = "Python syntax highlighting"
print s
```












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