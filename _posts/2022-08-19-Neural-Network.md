---
title: Artificial Neural Network
date: 2022-08-19 19:55:00 +0800
categories: [Machine Learning, Neural Network]
tags: [Machine Learning, Neural Network]
math: true
mermaid: true
---

인공신경망에 대해 알아본다. 먼저 수학적인 표현에 대해 살펴본 이후 아주 간단한 구조의 인공신경망에 Gradient Descent를 적용해본다. 정리된 알고리즘대로 Python으로 Machine Learning도 진행한다. 


## 1. Intro

Michael Nielsen의 ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/)은 인공신경망에 대해 공부하기 매우 좋은 자료다. Andrew NG 교수의 Coursera수업과 함께 참고하여 이 포스팅 작성한다. 

인공신경망은 이름에서 알 수 있듯이 인간의 신경계, 뉴런의 행동양식을 모방해서 만들어졌다는 이야기 들어봤을 것 같다. 반면 실제로 인간뉴런의 행동은 그리 간단하지도 않으며 이런 시각이 인공신경망 기법에 대한 정확한 이해를 방해한다는 의견도 있다.  

내 생각에도 인공신경망은 수학모델이고 Regression에 활용될 뿐이다. 다만 모든 모양의 함수를 표현할 수 있는 매우 큰 그릇같은 느낌이다. 실제로 수학적으로 그렇다. 증명은 여기서 안다루고 관련 자료는 [여기](http://neuralnetworksanddeeplearning.com/chap4.html) 참고하면 된다.

보통 [아래 그림](https://www.ibm.com/kr-ko/cloud/learn/neural-networks)처럼 인공신경망을 표현한다.

![Deel Learning](https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_01_03-DeepNeuralNetwork-WHITEBG.png){: width="500" height ="300"}

가장 왼쪽 Data가 들어가는 Input Layer에서 시작해서 가운데에는 뭔지 모르지만 여러겹의 Hidden Layer를 거쳐 가장 오른쪽의 Output Layer를 통해 결과가 나온다. 원으로 표현된 각 Unit들은 화살표로 빼곡하게 그물망처럼 연결되어 있다.

직관적인 느낌은 줄 수 있지만 아직 어떻게 작동하는지 정확히 모르겠다.

## 2. Mathematical Expression

아래 그림처럼 가운데 Hidden Layer를 모두 지우고 각 Layer에 Unit들은 한개씩 있는 인공신경망을 생각해본다. 


![ANN_config1](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-10/ANN_1.png){: width="500" height="300"}

위 그림 제일 왼쪽 Input Unit안에 x하나가 있다. x는 Input이고 변수(variable)이며 특성(feature)이라고도 부른다. 화살표를 따라 오른쪽으로 가면 Output이 나오며 모델을 통한 결과값으로 hypothesis를 의미하는 h로 표시되 있다. 

선을 따라 오른쪽으로 가다보면 $\theta_1$, $\theta_0$를 만나며 input인 x와의 선형 조합 과정을 거친다. Slope에 해당하는 $\theta_1$은 weight, intercept에 해당하는 $\theta_0$ bias라고 부른다. 편의상 이 중간 과정을 $a$로 표현하면 아래와 같다.

$$a = \theta_1 x + \theta_0 \qquad (1)$$

선형 조합 이후 순차적으로 활성화(activation)이라는 과정을 거쳐 최종 Output을 얻는다. Activation 함수는 여러가지 선택지가 있으며 이 포스팅에서는 [sigmoid 함수](https://ko.wikipedia.org/wiki/%EC%8B%9C%EA%B7%B8%EB%AA%A8%EC%9D%B4%EB%93%9C_%ED%95%A8%EC%88%98)를 사용하겠다.

$$g(x) = {1 \over {1 + {e}^{-x}}}\qquad (2)$$

Sigmoid 함수는 아래그림처럼 생겼다. x가 일정값 이하일때 함수값이 0에 가깝다가 일정값 이상이 되면 급격히 1에 가까워지는 성질로 활성화를 표현한다. 

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png){: width="300" height="300"}

결국 최종 h는 아래 식과 같이 표현된다. (여기까지는 [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)과 완전히 동일한 형태이다.)

$$ h_\theta (x) = g(a) = {1 \over {1 + {e}^{-(\theta_1 x + \theta_0)}}} \qquad (3)$$

정리해보면 
1. 각 Unit들을 이어주는 선은 $\theta$로 표현되는 weight, bias를 의미하고
2. 왼쪽에서 오른쪽으로 선을 따라가면서 순차적으로 선형조합(식(1))과 활성화(식(2))라는 과정을 거친다.
3. 최종 계산값은 선형조합과 활성화를 거친 식(3)으로 주어진다.

인공신경망을 이용한 머신러닝이라고 한다면 $\theta$로 표현되는 parameter들을(weight, bias) 데이터에 잘 맞게 regression하는 과정일 것이다.

좀 더 복잡한 구조로 Input, Output Layer에 Unit이 각각 2개인 경우를 생각할 수 있다. 

![ANN_config1](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-10/ANN_2.PNG){: width="400" height="300"}

변수는 $x_1$, $x_2$ 2종류, output도 $h_1$, $h_2$로 2개이며 아래 수식과 같이 정의된다.

$$h_{1, \theta}(x_1, x_2) = {1 \over {1 + {e}^{-(\theta_{01}+\theta_{11}x_1 + \theta_{21}x_2)}}} \qquad (4)$$

$$h_{2, \theta}(x_1, x_2) = {1 \over {1 + {e}^{-(\theta_{02}+\theta_{12}x_1 + \theta_{22}x_2)}}} \qquad (5)$$

각 문자들에 붙는 아래첨자에 숫자들이 늘어나서 좀 복잡해 보일 수 있다. $\theta_{ij}$는 i번째 변수에서 j번째 output에 관련된 weight를 나타내며 i가 0인경우 j번째 output에 관여하는 bias다. 간단히 생각하면 각각의 output unit에 모이는 변수, weight, bias를 선형조합하고 활성화 과정을 거치는 동일한 방법이다.

마지막으로 Hidden Layer를 한개만 추가한 경우를 표현해보자. 아래 그림에는 Input, Hidden, Output Layer에 Unit이 각각 2개씩 총 6개인 구조의 인공신경망이 있다.

![img](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-10/ANN_3.PNG){: width="500" height="400"}

편의상 hidden layer의 각 unit별 output을 $a$로 표현한다. 
$\theta_{ij}^{k}$는 왼쪽부터 k번째 layer에 있는 i번째 input에서 k+1번째 layer에 있는 j번째 output으로 향하는 weight를 나타내고 i가 0인 경우는 bias다. 말로 설명하니까 굉장히 이상한데, 달라진건 아무것도 없다. 동그란 unit으로모이는 화살표들에 관련된 변수와 parameter들을 선형조합하고 활성화하는 동일한 규칙이 적용된다.

중간단계에 있는 $a_1$, $a_2$는 아래와 같이 표현된다.

$$a_{1} = {1 \over {1 + {e}^{-(\theta_{01}^1+\theta_{11}^1x_1 + \theta_{21}^1x_2)}}} \qquad (6)$$

$$a_{2} = {1 \over {1 + {e}^{-(\theta_{02}^1+\theta_{12}^1x_1 + \theta_{22}^1x_2)}}} \qquad (7)$$

잘 보면 식 (4), (5)와 동일하다. 동일한 규칙, 동일한 구조로 동일한 결과가 나온다. 최종 output인 $h_1$과 $h_2$는 아래와 같다.

$$h_{1} = {1 \over {1 + {e}^{-(\theta_{01}^2+\theta_{11}^2a_1 + \theta_{21}^2a_2)}}} \qquad (8)$$

$$h_{2} = {1 \over {1 + {e}^{-(\theta_{02}^2+\theta_{12}^2a_1 + \theta_{22}^2a_2)}}} \qquad (9)$$

만일 hidden layer가 1개층이 아닌 여러개의 층으로 구성되어 있다면 $a_i$였던 것들을 $a_i^j$로 j번째 layer에 있는 i번째 unit 처럼 표현해줄 수 있겠다.

절대 식 하나하나 기억할 필요 없다. 한번정도 눈으로 꼼꼼히 따라가보면 좋겠지만 그마저도 크게 필요없다. 오직 기억 할만한 것은 인공신경망은 **Input과 Parameters의 선형조합과 이후 활성화 과정으로 이루어진다**는 점이다. 

### 2.1. Hyper-parameter

weight와 bias처럼 선형조합에 사용되는 일종의 coefficient들을 parameter라고 불렀다. 추가로 hidden layer의 layer층 개수를 특별히 hyper-parameter라고 부른다. 인공신경망의 구조에 관련된 숫자라는 의미다. Data의 크기를 포함한 다양한 이유로 hidden-layer층 개수는 많을수도, 적을수도 있다. 관련한 자세한 내용이 궁금하면 [이 문서](https://ikkison.tistory.com/92)가 좋아보인다.

Parameter는 Gradient-Descent같은 방식으로 구하지만 hyper-parameter는 그러지 못한다. 경험적으로 구하거나 trial-error를 해야하며 정형화된 방법은 없는 것 같다. 그냥 일단 해보고 결과보고 그에따라 대응하는 것 같다.

### 2.2. Vectorization

식들을 잘 보면 덧셈이 많다. 위 식들은 Feature vector와 parameter vector들의 dot product, 내적으로 표현하면 여러번 반복해서 써야하는 수고로움을 덜 수 있다. 예를 들어 아래와 같은 단순합을 vector를 이용해서 간단하게 표현해보자.

$$S = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + ... + \theta_n x_n$$

$$ Let \quad \vec \theta = [\theta_0, \theta_1, ... , \theta_n], \quad \vec x = [1, x_1, x_2, ..., x_n]^T$$

$$ S = \vec \theta \cdot \vec x$$

bias에 대응될 변수에는 1을 넣어주고 dot product 하면 끝이다. 문자로 표현해서 인공신경망의 크기나 구조에 상관없이 일반화된 식을 사용할 수도 있다. 자주 얘기하는 Andrew NG교수 4주차 수업에서 vectorization을 활용해 인공신경망을 표현하는 것을 볼 수 있다. 

굳이 이 얘기를 하는건 코드를 짤 때도 vectorized 형태로 연산을 정의하면 단순 반복문의 경우보다 계산속도가 훨씬 빠르기 때문이다. 단순 for문 보다 numpy array같은거 이용해서 계산하는게 훨씬 빠르긴 하다. 수치연산 관련 코드가 vector나 행렬 형태에서 좀 더 최적화되어 움직이도록 되어 있는 것 같다. 나도 잘 몰라서 더 얘기하기는 힘들다.

지금은 크게 신경쓰지 않아도 좋다. 어차피 이런거 신경쓸 정도의 무거운 계산을 하기까지 앞으로 멀기도 했고 그때 쓰는 라이브러리에서 이미 최적화된 방식으로 잘 해줄 것이다.

## 3. Gradient Descent

앞에서 얘기한 Input에서 Output까지 가는 일련의 계산 흐름을 FeedForward라고 부른다. 반대로 가는 과정을 Back-Propagation이라고 부르며  Cost를 최소화하는 parameter들을 구하는 과정을 말한다.

수식을 간단하게 하기 위해서 hidden layer없이 feed unit 2개, output unit 1개짜리 인공신경망을 만들고 Gradient Descent를 적용해 본다. 

![img](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-10/ANN_4.PNG){: width="400" height="300"}

Output $h$는 다음식으로 표현된다.

$$h = {1 \over {1 + {e}^{-(\theta_0 + \theta_1 x_1 + \theta_2 x_2)}} } \qquad (10)$$

연습삼아 Vectorized Form으로 표현 해보자. x vector안에 $x_0$가 추가된 것에 주의해야한다. 변수를 추가한 것은 아니며 단순히 bias를 받아주기 위한 처리에 불과하다. ($x_0$ 값은 모두 1이다.)

$$ \vec x = [x_0(=1), x_1, x_2]^T,\quad \vec \theta = [\theta_0, \theta_1,\theta_2] $$

$$h_{\vec \theta}(\vec x) = {1 \over {1 + {e}^{-(\vec \theta \cdot \vec x)}} }  = g(\vec \theta \cdot \vec x)$$

모델값과 Data간의 오차의 합을 Cost라고 하고 아래식으로 주어진다.

$$ J(\vec \theta) = {1 \over{2m}} \sum_{i=1}^m (h_{\theta}(x^i)-y^i)^2  \quad \quad (11) $$

목표는 Cost를 최소화하는 parameter vector $\vec\theta$ 를 구하는 것이다. [Gradient Descent](https://ch-hey.github.io/posts/Gradient-Descent/)에 따르면 이 때 Parameter Update 방식은 아래 식으로 주어진다. (12a, b는 동일한 표현이다.)

$$ \theta_i  := \theta_i - \alpha {\partial \over \partial {\theta_i}} J(\vec \theta) \quad (i=0,1,2)  \qquad (12a) $$

$$\Delta \vec \theta = -\alpha\nabla J(\vec \theta) \qquad  (12b)$$

남은 일은 편미분하는 일이다.

$${\partial \over \partial {\theta_i}}\left({1 \over{2m}} \sum_{i=1}^m (h_{\theta}(x^i)-y^i)^2\right) = {1 \over {m}}\sum_{i=1}^m (h_{\theta}(x^i)-y^i){\partial \over \partial {\theta_i}}h_{\theta}(x)$$

$$ = {1 \over {m}}\sum_{i=1}^m (h_{\theta}(x^i)-y^i){\partial \over \partial {\theta_i}}g(\vec \theta \cdot \vec x)$$

$$ = {1 \over {m}}\sum_{i=1}^m (h_{\theta}(x^i)-y^i)g'(\vec \theta \cdot \vec x) x_i$$

sigmoid 함수의 미분식은 아래와 같이 정리된다.

$$g(x) = {1 \over {1 + {e}^{-x}}} \quad g'(x) = {-{e}^{-x} \over {(1+{e}^{-x})^2}} = g(x)(1-g(x))$$

따라서 식 12a, b는 아래식처럼 정리된다.

$$\theta_i  := \theta_i - \alpha {1 \over {m}}\sum_{i=1}^m (h_{\theta}(x^i)-y^i)g(\vec \theta \cdot \vec x)(1-g(\vec \theta \cdot \vec x)) x_i \qquad (13)$$


좀 더 복잡한 구조에 대한 일반화된 식과 이 구조에서의 Back-propagation에 대해 수식으로 유도하고 정리된 자료를 보고싶다면 Michael Nielsen의 "Neural Networks and Deep Learning"책의 이 [Chapter](http://neuralnetworksanddeeplearning.com/chap2.html) 정독해보자. Notation은 좀 다르지만 원하는 것을 얻을 수 있다.

## 4. Application

필요한 식 정리가 끝났다. 데이터는 아래 정리된 표를 사용할 것이다. 식 유도에 사용된 인공신경망 구조에 맞게 Input의 종류는 2가지, Output은 1가지이다.

|$x_1$|$x_2$|Output|
|:---:|:---:|:---:|
|0|0|0|
|1|0|1|
|0|1|1|
|1|1|1|

표를 잘 보면 $x_1$, $x_2$ 둘 중 하나만 1이어도 결과값이 1이고 두개 모두 동시에 0일때만 결과값은 0인 관계임을 알 수 있다. 이런 관계를 logical OR function이라고 한다. (보통 0은 거짓, 1은 참을 의미하며 OR연산에서는 둘 중 하나만 참이어도 결과는 참이다.)

달리 표현하면 머신러닝을 활용해서 인공신경망으로 logical OR function을 만들어 볼 것이다. (실은 logistic regression이다.)

필요한 라이브러리를 Import 해주자. Random은 parameter들의 초기값을 정하는데 써준다.

```python
import numpy as np
import random, os
import matplotlib.pyplot as plt
```
위 표에 있는 데이터 입력한다. x0는 1에 해당하며 bias 부분을 표현해 주기 위해 쓰인다.

```python
weights = list()

for k in range(3):
    weights.append(random.random())

weights = np.array(weights)

x0data = [1, 1, 1, 1] # x0 = 1
x1data = [0, 1, 0, 1] # x1
x2data = [0, 0, 1, 1] # x2
x = np.array([x0data, x1data, x2data])

ydata = [0, 1, 1, 1] # Output
```

Learning rate는 학습을 몇 번 돌려보면서 정했다. 총 1000번 학습(parameter update)가 이루어지게 설정했다. `err_list`는 학습 횟수가 늘어남에 따라 Cost가 어떻게 변하는지 보기 위해 쓰인다.

```python
alpha = 40
err_list = []

def sigmoid(x):
    return 1/(1+np.exp(-x))

# m번째 data와 weights간 선형조합
def lincomb(m):
    return np.dot(weights, x[:, m])

# m번째 data에서 계산한 모델값과 데이터간 차이
def err(m):
    return (sigmoid(lincomb(m)) - ydata[m])

for j in range(1000):

    grad_x0 = 0
    grad_x1 = 0
    grad_x2 = 0
    err_loc = 0

    for i in range(len(x1data)):
        grad_x0 = grad_x0 + err(i)*sigmoid(lincomb(i))*(1-sigmoid(lincomb(i)))*x0data[i]
        grad_x1 = grad_x1 + err(i)*sigmoid(lincomb(i))*(1-sigmoid(lincomb(i)))*x1data[i]
        grad_x2 = grad_x2 + err(i)*sigmoid(lincomb(i))*(1-sigmoid(lincomb(i)))*x2data[i]
        err_loc = err_loc + (err(i)**2)

    err_list.append(err_loc)

    weights[0] = weights[0] - alpha * grad_x0 * (1/len(x1data))
    weights[1] = weights[1] - alpha * grad_x1 * (1/len(x1data))
    weights[2] = weights[2] - alpha * grad_x2 * (1/len(x1data))
```


```python
print(weights)
for i in range(len(x1data)):
    print(f"(x1={x1data[i]}, x2={x2data[i]}) => y={ydata[i]}, Model Output = {sigmoid(lincomb(i)):.2f}")


plt.plot(err_list)
plt.title("Error During Learning")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()

```

    [-4.1032397   8.67424375  8.67424407]
    (x1=0, x2=0) => y=0, Model Output = 0.02
    (x1=1, x2=0) => y=1, Model Output = 0.99
    (x1=0, x2=1) => y=1, Model Output = 0.99
    (x1=1, x2=1) => y=1, Model Output = 1.00
    


    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-10/output_3_1.png)
    
위 결과를 보면 데이터와의 오차가 충분히 작아졌다고 판단된다.


```python
x_plot_whole = np.arange(-0.25, 1.25, 0.1)
x_plot_1 = np.array([1, 0, 1])
y_plot_1 = np.array([0, 1, 1])

x_plot_0 = np.array([0])
y_plot_0 = np.array([0])

y_line = (1/weights[1])*(-weights[2]*x_plot_whole - weights[0])

plt.scatter(x_plot_1, y_plot_1, marker = 's')
plt.scatter(x_plot_0, y_plot_0, marker = 's')
plt.plot(x_plot_whole, y_line, 'r')
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

```

(x1, x2, y) 형태의 data를 2d로 표현해보면 아래 그림과 같다. 파란색 점은 y=1인 경우고 주황색 점은 y=0인 경우에 해당한다. 빨간 선은 학습결과 구해진 weight를 가지고 선형조합 부분을 표현한 것이며 x1과 x2로 이루어진 plane을 y=0과 y=1이 나오는 부분으로 나눠주는 것을 볼 수 있다.

빨간색 선 아래 부분은 활성화되지 못하는 영역, 빨간색 선 위 부분은 활성화되는 영역이라고 생각해도 될 것 같다. 빨간선을 Decision Boundary라고 부르기도 한다.
    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-10/output_4_0.png)
    
## Summary

인공신경망의 작동방식을 확인해 봤다. 간단히 말하면 Features와 Parameter의 선형조합 이후 활성화 과정으로 이어지는 순차적인 연산이었다. 

실은 hidden layer가 없는 인공신경망은 결국 logistic regression과 동일한 구조라서 인공신경망을 했다고 말하기는 좀 민망하긴 하다. 하지만 작동원리를 알아보는데는 이정도면 충분할 것 같다. 결국 서로 복잡하게 얼키고 설킨 logistic regression이 인공신경망이라고 생각한다.

수식은 나름 정리해봤지만 정말 간단한 케이스에 대해서만 봤을 뿐이다. 진짜 수학적으로 궁금한게 많다면, Michael Nielsen의 ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/) 다시 한번 추천한다.

실은 수식은 다 까먹어도 상관없다. 다시는 Gradient Descent도 볼 일 없을 것이다. 이런게 있었지 하는 정도로 충분하다. 위에 보이는 하드코딩도 더이상 안해도 될 것같다. 이제 실제 꽤 큰 데이터로 머신러닝을 진행해 볼 것이다. 다음 포스팅은 [Tensorflow Regression Tutorial](https://www.tensorflow.org/tutorials/keras/regression)예제를 중심으로 해보려고 한다.