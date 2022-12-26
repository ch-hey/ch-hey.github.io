---
title: "[머신러닝] Tensorflow"
date: 2022-12-25 19:55:00 +0800
categories: [Machine Learning]
tags: [Deep Learning, Tensorflow]
math: true
mermaid: true
---

![tensorflow logo](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/TensorFlow_logo.svg/1280px-TensorFlow_logo.svg.png){:. width="400"}


Tensorflow로 Deep-Learning 해보자. [Tensorflow Blog](https://www.tensorflow.org/?hl=ko)에 잘 정리된 튜토리얼 따라할거니까 생각보다 어렵지 않다. 

물론 제대로 하려면 여러가지를 알아둬야 한다. 인공신경망, 딥러닝, 여러 수학들이랑 정의들, 라이브러리 테크닉들, 그리고 제일 중요한건 딥러닝을 적용해보려는 분야에 대한 지식이다. 이건, 어떻게 할 수 없으니 알아서 잘 해보자.

좀 더 원활한 이해를 위해서는 [여기](https://ch-hey.github.io/posts/AI-Definition/)와 [여기](https://ch-hey.github.io/posts/Neural-Network/)를 읽어보고 시작하기를 추천드린다. 아니면 이 포스팅은 그냥 가볍게 읽으면서 지나가도 괜찮을 것 같다.


## 1. Intro

딥러닝, 인공신경망 이거는 다른 [포스팅](https://ch-hey.github.io/posts/AI-Definition/)에 정리되어 있고, Tensorflow에 대해 간단히 알아본다. [위키](https://namu.wiki/w/TensorFlow?from=Tensorflow)에서는 이렇게 말한다.

> 구글이 2011년에 개발을 시작하여 2015년에 오픈 소스로 공개한 기계학습 라이브러리.
{: .prompt-info}

일단 구글에서 만들었고, 잘은 모르지만 Python만으로 만들어진건 아니고 계산속도 빠른 C같은 언어들도 섞여있다고 한다. 당연히 학습 속도가 빠르고 조금 제한이 있지만 GPU연산도 지원한다고 한다.

보통 구글링으로 딥러닝을 검색해보면 많이 나오는게 이미지 인식, 자연어 처리 같은 예제 들이고 이런 곳에서 쓰이는 방법들(convolutional, recurrent 류들) 구현이 코드 몇 줄로 가능하다. 이 외에도 효율적인 학습을 위한 다양한 계산 옵션들, 콜백들(early stopping, dropout 등)을 제공한다.

완전 커스터마이징이 가능한 건 아닌 것 같지만, 상당한 수준의 모델빌드가 생각보다 쉽게 가능하다. Teonsorflow Blog에 튜토리얼이 잘 정리되어 있어서 그대로 따라하면 기능구현을 어렵지 않게 따라해 볼 수 있다.

## 2. TensorFlow Tutorial - Regression

이미지나 자연어에 관한 딥러닝에 대해서는 실은 크게 관심이 없다. 일하면서 별로 겪을 수 있는 일들도 없고, 특히 데이터를 미리 정제하고 준비시키는게 조금 짜증나서 별로 재미 없었다. 물론, 딥러닝을 잘 이해하려면 이런 예제들 한 번 정도 겪어보는게 나쁘진 않은 것 같다. 확실히 인기는 많은지 구글링 해보면 대부분 나오는 예제들이 이 두 가지다.

여기서는 단순한 Regression 예제를 Deep Learning으로 해본다. 숫자를 넣고, 숫자가 나온다. 물론, 이미지나 자연어 처리도 숫자를 넣고 숫자가 나오기는 매한가지긴 하다.

## 3. 자동차 연비 예측하기: 회귀

이 [튜토리얼](https://www.tensorflow.org/tutorials/keras/regression?hl=ko)은 개인적으로 정말 좋아하는 예제다. 이미지나 자연어처리만 넘쳐나는 딥러닝 예제에서 얼마안되는 단순회귀 관련 문제이다. 자동차 무게, 배기량, 실린더수, 마력, 연식 등의 특성에 따른 자동차의 연비 데이터를 활용해서 자동차 연비 예측하는 딥러닝 모델을 만들거다. 그리고 이 방법을 조금만 잘 이용하면 여러가지 일들을 시도해볼 수 있다. 

먼저 이 예제를 실행하려면 Python 라이브러리 중 Tensorflow, Seaborn, Pandas가 필요하다. 없거나 처음 들어봤다면 설치해준다. 

### 3.1. Data 준비 및 전처리

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

Tensorflow는 2.11.0을 썼다. Tensorflow 내 keras에서 자동차연비 데이터를 불러온다. 

```python
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path
```

```python
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MPG</th>
      <th>Cylinders</th>
      <th>Displacement</th>
      <th>Horsepower</th>
      <th>Weight</th>
      <th>Acceleration</th>
      <th>Model Year</th>
      <th>Origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>393</th>
      <td>27.0</td>
      <td>4</td>
      <td>140.0</td>
      <td>86.0</td>
      <td>2790.0</td>
      <td>15.6</td>
      <td>82</td>
      <td>1</td>
    </tr>
    <tr>
      <th>394</th>
      <td>44.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>52.0</td>
      <td>2130.0</td>
      <td>24.6</td>
      <td>82</td>
      <td>2</td>
    </tr>
    <tr>
      <th>395</th>
      <td>32.0</td>
      <td>4</td>
      <td>135.0</td>
      <td>84.0</td>
      <td>2295.0</td>
      <td>11.6</td>
      <td>82</td>
      <td>1</td>
    </tr>
    <tr>
      <th>396</th>
      <td>28.0</td>
      <td>4</td>
      <td>120.0</td>
      <td>79.0</td>
      <td>2625.0</td>
      <td>18.6</td>
      <td>82</td>
      <td>1</td>
    </tr>
    <tr>
      <th>397</th>
      <td>31.0</td>
      <td>4</td>
      <td>119.0</td>
      <td>82.0</td>
      <td>2720.0</td>
      <td>19.4</td>
      <td>82</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

자동차 무게, 배기량, 실린더수, 마력, 연식 등의 특성에 따른 자동차의 연비가 테이블로 보인다. Dataset의 크기는 397개다.

isna()는 결측값이 있는지 확인하는 함수다. 아래에서 보면 Horsepower부분에 6개의 결측치가 있는 것으로 보인다. 유저가 확인해서 채워넣을 수도 있겠지만 보통 이런부분은 그냥 통째로 날린다. Pandas에 dropna()라는 편한 함수가 있다.


```python
dataset.isna().sum()
```




    MPG             0
    Cylinders       0
    Displacement    0
    Horsepower      6
    Weight          0
    Acceleration    0
    Model Year      0
    Origin          0
    dtype: int64




```python
dataset = dataset.dropna()
```

Origin은 제조 장소를 의미한다. 숫자 1, 2, 3으로 되어 있고 각각 미국, 유럽, 일본에 해당한다. 머신러닝에 넣어주는 데이터는 텍스트를 넣을 수는 없으므로 보통 이런 식으로 숫자로 표현한다.

Origin은 categorical이라고 말하고 범주형이라고도 표현하며 연속적인 값을 가지지 않는다. 그래서 여기서는 미국, 유럽, 일본의 특성을 만들고 각 지역에 해당하면 1(True)을 넣고 아니면 0(False)을 넣는 식으로 데이터를 처리한다.

데이터를 이런 식으로도 표현하는구나 하고 넘어가면 언젠가 쓰일 것 같다.

```python
origin = dataset.pop('Origin')
```


```python
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
```

Dataset을 훈련 세트와 테스트 세트로 분할한다.

이 부분에서는 예제들마다 약간씩 설정이 다른데, training, validation, test set로 3개로 6/2/2로 분할하기도 한다. 분할 비율이나 데이터를 나누는 방법도 상황마다 다르기도 하다. 여기는 특별히 정답은 없는 것 같다.

training은 말 그대로 인공신경망 모델내 파라미터를 피팅하는데 쓰이고 validation은 피팅된 파라미터를 가진 모델을 1차적으로 평가하는데 쓰인다. 이때 훈련성과가 별로 좋지 않다면 모델의 구조를 바꾸거나 새로운 특성을 추가하거나 하는 일들을 한다.

마지막으로 training과 validation data에 적당한 fitness를 보여주는 모델을 최종 확인해 볼 때 test dataset을 활용한다. 어떠한 학습과정에서도 사용되지 않은 정말 깨끗한 데이터를 통해 확인한다는 의미 인 것 같다.

어찌됬든, 여기서는 위의 설명에 있는 용어의 의미를 따르자면 test set는 없고, training과 validation 이렇게 두개의 세트로 데이터를 분할한다. 분할비율은 8/2다. 7/3이나 6/4도 뭐, 상관은 없을 것 같다.


```python
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
```
밑에서는 seaborn 라이브러리에서 pairplot이라는 것으로 각 데이터 특성간의 영향을 확인한다. 두 특성들이 주고받는 영향은 파악할 수 있겠다. 복잡한 경우는 하나마나겠지만.

```python
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
```
    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-12-25/output_8_1.png)
    


데이터 특성들의 통계적 수치들도 확인한다. Pandas 라이브러리에서 describe() 함수를 쓴다. 요거는 그래도 쓸만해 보인다.

```python
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cylinders</th>
      <td>314.0</td>
      <td>5.477707</td>
      <td>1.699788</td>
      <td>3.0</td>
      <td>4.00</td>
      <td>4.0</td>
      <td>8.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Displacement</th>
      <td>314.0</td>
      <td>195.318471</td>
      <td>104.331589</td>
      <td>68.0</td>
      <td>105.50</td>
      <td>151.0</td>
      <td>265.75</td>
      <td>455.0</td>
    </tr>
    <tr>
      <th>Horsepower</th>
      <td>314.0</td>
      <td>104.869427</td>
      <td>38.096214</td>
      <td>46.0</td>
      <td>76.25</td>
      <td>94.5</td>
      <td>128.00</td>
      <td>225.0</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>314.0</td>
      <td>2990.251592</td>
      <td>843.898596</td>
      <td>1649.0</td>
      <td>2256.50</td>
      <td>2822.5</td>
      <td>3608.00</td>
      <td>5140.0</td>
    </tr>
    <tr>
      <th>Acceleration</th>
      <td>314.0</td>
      <td>15.559236</td>
      <td>2.789230</td>
      <td>8.0</td>
      <td>13.80</td>
      <td>15.5</td>
      <td>17.20</td>
      <td>24.8</td>
    </tr>
    <tr>
      <th>Model Year</th>
      <td>314.0</td>
      <td>75.898089</td>
      <td>3.675642</td>
      <td>70.0</td>
      <td>73.00</td>
      <td>76.0</td>
      <td>79.00</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>314.0</td>
      <td>0.624204</td>
      <td>0.485101</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>314.0</td>
      <td>0.178344</td>
      <td>0.383413</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>314.0</td>
      <td>0.197452</td>
      <td>0.398712</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

데이터에서 input 특성들과 output 특성(자동차 연비, MPG)을 분리한다. 딥러닝 모델 학습의 input으로 쓰일 dataset은 train_labels에 해당한다. 

```python
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
```

데이터 정규화라는 것을 해본다. Dataset의 특성들(feature, 혹은 input)의 스케일과 범위가 다르면 normalization, 정규화를 하는 것이 추천된다고 한다. 특성을 정규화하지 않아도 모델은 수렴하지만 훈련시키기 어렵거나 입력단위에 의존하는 모델이 만들어 질 수 있다고 한다.

정규화는 수학적으로는 평균 0, 표준편차 1을 갖는 분포를 갖도록 데이터세트를 조정하는 작업이다. Standard Normal Distribution, 표준정규분포로 만드는 작업으로 기억안나면 구글링 해보자. 


```python
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
  
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
```

여기까지 데이터 전처리에 대해 정리해보자.

- 데이터를 준비한다.
- 데이터 내 결측치가 있는지 확인하고, 처리한다. (채워넣거나 싹 다 지운다.) 
- 범주형 데이터의 경우 적절하게 수치형으로 변환해준다.
- 또는 데이터에 noise가 있는 경우 적당한 filter 통한 처리들도 고민해 볼 수 있다.
- 특성들의 통계치들을 확인하고, 전체 dataset을 정규화 해준다.
- 전체 데이터를 training, validation, test set로 적절히 분할해준다.


### 3.2. 모델 빌드와 학습

드디어 딥러닝 모델을 빌드하고 학습시켜보자. 

아래 코드에서 build_model은 node가 64개씩있는 hidden layer가 2개짜리 완전연결(densely connected) 인공신경망 모델을 만들어준다. hidden layer의 개수나 node의 개수같은 hyper parameter를 정하는 것은 해보면서 바꿔봐야 할거같다. 정답은 없다. 이거 도와주는 tensorflow 내 기능이 있는것 같은데 [여기](https://www.tensorflow.org/tutorials/keras/keras_tuner?hl=ko) 한번 보자.

활성함수로 [relu](https://ko.wikipedia.org/wiki/ReLU)가 쓰인다. 다른 [포스팅](https://ch-hey.github.io/posts/Neural-Network/)에서 활성함수로 sigmoid를 사용한 다른 예제가 있다.

`keras.Sequential`내에서 `layer.Dense`로 하나씩 layer를 추가한다. input layer는 딱히 정의하지 않으며 그 다음 부터 정의한다. output layer를 보면 활성함수가 정의되어 있지 않다. 이전 layer에서의 선형조합만으로 결과값을 가지겠다는 의미로 모든 범위의 실수가 결과값으로 나올 수 있다.  

optimizer는 RMSprop라는 것을 쓴다고 한다. 어떤 수학적인 기법인지 알아보면 좋겠지만, 일단 넘어가자. 인공신경망 모델 내 파라미터를 피팅하는 수학적인 알고리즘중 하나다. Gradient Descent방식을 쓸 수도 있다. SGD (Stochastic Gradient Descent)으로 옵션을 주면 된다. 예제는 [여기](https://codetorial.net/tensorflow/basics_of_optimizer.html) 한 번 보자.

loss로 'mse'를 지정해 준 것은 mean squared error를 의미하며 보통 오차함수로 많이쓰는 오차제곱평균이다. 'mae'등 다른 옵션들도 가능하다. 커스터마이징된 오차함수를 정의하는 것도 가능한 것 같은데, 필요하면 각자 찾아보자.


```python
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
        ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model
```


```python
model = build_model()
```

파라미터의 개수를 보면서 내가 만드는 인공신경망 모델의 구조를 파악해 볼 수도 있다.

준비한 dataset의 input 특성들의 종류는 9개다. 각 hidden layer별 node개수는 64개로 정했다. input layer와 첫 번째 hidden layer간의 파라미터 개수는 bias 64개와 weight 9 $\times$ 64 = 576개로 총 640개다.

첫 번재 hidden layer와 두 번째 hidden layer간의 파라미터 개수는 bias 64개와 weight 64 $\times$ 64 = 4096개로 총 4160개다.

마지막으로 두 번째 hidden layer와 마지막 ouput layer간의 파라미터 개수는 bias 1개와 weight 64 $\times$ 1 = 64개로 총 65개다.


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 64)                640       
                                                                     
     dense_1 (Dense)             (None, 64)                4160      
                                                                     
     dense_2 (Dense)             (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 4,865
    Trainable params: 4,865
    Non-trainable params: 0
    _________________________________________________________________
    


모델 학습이 진행되고 있는지 확인하는 PrintDot이라는 콜백 함수를 정의한다. 학습회수 100번마다 점하나씩 프린트 한다고 한다. `model.fit()`으로 드디어 모델 학습을 수행하는 명령을 내린다. history라는 변수에 모델 학습 내역들이 기록된다.


```python
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[PrintDot()])
```

    
    ....................................................................................................




```python
import matplotlib.pyplot as plt

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(8,12))

    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
            label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
            label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()

plot_history(history)
```
파란색의 training dataset에 대한 오차는 Epoch, 학습 회수가 증가할 수록 줄어드는 것을 확인할 수 있다. 하지만 주황색의 validation datset에 대한 오차는 처음에는 좀 줄어들었다가 학습회수가 증가하면서 오히려 늘어난다.

overfitting이라고도 부르는 이런 현상은 딥러닝 모델이 training dataset에만 맞도록 지나치게 fitting되면서 오히려 전체적인 dataset에 대한 general한 특성을 잃게되는 것을 의미한다.

    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-12-25/output_18_0.png)
    

여러가지 해결법이 있겠지만 여기서는 early stopping, 학습을 조기종료하는 방식으로 이 문제를 해결해보려는 것 같다. 아래 `keras.callbacks.EarlyStopping`에서 validation dataset에 대한 오차를 계속 모니터하면서 10번이 지나도 이전보다 학습성능이 좋아지지 않으면 강제로 멈추겠다는 설정을 해준다.


```python
model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)
```

    
    ............................................................


    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-12-25/output_19_1.png)


모델 빌드와 학습도 적당히 정리해보자.

- 모델을 정의한다. 여기서는 완전연결 인공신경망, hidden layer2개짜리 딥러닝 모델이다. Hyper parameter는 일단 해보고 바꿀 생각으로 정한다.
- optimizer방식과 오차함수 형태 등의 세부 옵션을 설정한다.
- 일단 학습 진행해 보면서 Overfitting 혹은 underfitting여부를 확인한다.
- overfitting이면 early stopping등의 콜백옵션들을 생각해본다.
- underfitting이면 특성의 종류를 늘리거나 모델의 구조를 더 복잡한 것으로 (파라미터를 늘리는 방향으로) 변경해 볼지 생각해본다.

학습횟수에 따른 오차의 변화를 확인해보면서(learning curve를 본다고도 한다.) 고생해서라도 데이터를 더 추가해야할지, 모델을 어느 방향으로 변경할 지, 학습의 세부 옵션을 어떻게 변경할지 등등의 대응 방식들을 고민해야 하는 것 같다.

자세한 건, 따로 정리할 기회가 있으면 좋겠지만, 나도 잘 모르기 때문에 강의나 책을 찾아보길 추천한다. Coursera 한 번 빡세게 듣거나 여러 유튜브 보면 좋을 것 같다.


### 3.3. 모델 평가


```python
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("테스트 세트의 평균 절대 오차: {:5.2f} MPG".format(mae))
```

    3/3 - 0s - loss: 5.6281 - mae: 1.8268 - mse: 5.6281 - 19ms/epoch - 6ms/step
    테스트 세트의 평균 절대 오차:  1.83 MPG
    


```python
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
```

    3/3 [==============================] - 0s 977us/step
    


    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-12-25/output_21_1.png)
    



```python
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
```


    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-12-25/output_22_0.png)
    

학습이 완료된 모델을 이용해 예측하고 평가한다. 오차에 대한 히스토그램을 보면 어디 한 군데에 집중되지 않고 적당한 분포를 보인다.

모델을 저장하고 불러오는 부분은 이 예제에 없어서 [여기](https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko)를 보면 될 것 같다.

HDF5라는 표준 기본 저장 포맷이 있다고 한다. 아래 명령어로 저장하고 불러오면 될 것 같다.

```python
# 새로운 모델 객체를 만들고 훈련합니다
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# 전체 모델을 HDF5 파일로 저장합니다
# '.h5' 확장자는 이 모델이 HDF5로 저장되었다는 것을 나타냅니다
model.save('my_model.h5')
```

```python
new_model = tf.keras.models.load_model('my_model.h5')
```

## 4. Application & Limitations

자동차 연비가 배기량이나 실린더수나 자동차 무게에 관련이 있을거 같긴 하지만 이를 연결해주는 어떤 이론식같은건 아마도 존재하지 않을 것 같다.

이런 상황들은 종종있다. 주가같은게 환율이나 물가상승률, 기준금리, 회사의 성장률, 분기별 영업이익 뭐 이런 것들에 영향이 있을 것 같지만 어떤 식으로 표현하기는 애매하다. 어떤 석유화학 공장의 생산품이 반응기의 온도, 압력, 조성, 수위, 제립기 온도 뭐 이런 것들에 영향을 받는건 분명한데 이를 어떤 식으로 표현하기는 어렵다.

이런 때 자동차 연비예측 예제를 꺼내들고 적당히 써보면, 잘만드는건 다른얘기지만 일단 뭐라도 만들어볼 수 있다. 

하지만 개인적으로는 인공신경망을 써봐도 현업에 빡세게 적용하기는 아직은 좀 이른감이 있다고 생각한다. 예측 정확도 적당한 모델은 얼마든지 만들어 볼 수 있다. 90% 넘는건 또 다른 얘기이긴 하지만. 근데 이걸 실제로 써먹는건 또 다른 차원의 얘기다. 이게 일이라는게, 그냥 재밌어보여서 했어요 할수있는 것도 아니고 일정 부분의 성과가 분명히 필요한데, 모델의 정확도는 training에 사용되지 않은 새로운 패턴에 대해 어디로 튈지 예측도 안되고 설명도 안되고 불안하다.

개인적인 생각일 뿐이고 관련해서 많은 경험들이 있는것도 아니다. 다른 많은 잠재적인 활용분야가 있는데 내가 모르는 것일 수도 있다. 아직은 그냥 모르는 것들이 너무 많다.

일단 여전히 뜨고있고 좋은 기술임에는 확실하니 나름 준비하고 공부는 한다. 취미로 해보기엔 아직은 재미있는 부분들이 있다.
    
## Summary

Tensorflow을 써서 딥러닝 모델을 만들고 학습도 시켜봤다. 튜토리얼 그대로 베껴서 똑같이 따라하는거지만, 알아야 할 것들이 꽤 많았다. 그리고 모델을 학습시키는 것보다도 훨씬 더 많은 단계들이 필요했던 부분은 바로 데이터 전처리였다. 데이터가 좋아야 모델도 좋다. Garbage in, garbage out. 좋은 데이터를 만들어주는 최일선에 항상 감사한 마음을 가져야 한다.

이런 얘기들 제일 마지막에 항상 하는 얘기는, 실은 딥러닝이니 머신러닝이니 인공신경망이니 이런거 보다도 적용하려는 분야에 대한 유저의 전문성이 가장 중요하다는 점이다. Input으로 넣는 특성이 output을 예측하는데 쓸데없는 것들로만 이루어져 있다면 딥러닝 할아버지가 오더라도 어떻게 해볼 수가 없이 시간낭비다. 이게 갖추어진 다음이 머신러닝 이론들, 수학, 코딩스러운 테크닉들 이런 것 같다.

다음은 scikit-learn을 해보던지 아니면 coursera 강의 좀 정리해보려 한다.

한해가 다 갔다. 그래도 한 달에 포스팅 하나씩은 썼다!!