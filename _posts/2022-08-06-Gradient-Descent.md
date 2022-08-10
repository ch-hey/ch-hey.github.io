---
title: Gradient Descent
date: 2022-08-08 19:55:00 +0800
categories: [Machine Learning, Gradient Descent]
tags: [Machine Learning, Gradient Descent, Linear Regression]
math: true
mermaid: true
---

## Intro

머신러닝에서 사용되는 알고리즘중 하나인 Gradient Descent, 경사하강법에 대해서 간단히 작성한다. 가능한 Andrew NG 교수의 Notation을 따르려고 한다.

머신러닝 중에서 답이 함께 있는 데이터를 사용하는 Supervised learning에 해당하며 단순 Linear Regression에 대한 예제로 설명한다. Data는 대표적인 Toy dataset인 iris를 이용한다. 

## Background

아래와 같이 변수 x에 대한 결과값 y에 대한 Dataset이 있고 우리는 이 둘 사이의 관계를 설명해주는 모델을 만들어야 한다. 모델이 만들어지면 데이터에는 없던 새로운 x를 넣었을 때 결과값인 y를 예측해 줄 것이다. 

$$(x^1, y^1)$$

$$(x^2, y^2)$$

$$(x^3, y^3)$$

$$...$$

$$(x^m, y^m)$$

데이터쌍 (x, y)의 개수는 m개 이다. 

데이터를 Plot 해봤는데 선형인 관계가 보였다면 자연스럽게 아래와 같이 선형식을 통해 모델을 만들려고 시도할 것이다.


$$h_{\theta} = {\theta}_0 + {\theta}_1 x \quad \quad (1) $$


h는 hypothesis를 나타내며 parameter로 ${\theta}_0, {\theta}_1$을 쓰는 선형식이다. 

이제 문제는 맨 위의 Dataset에 가장 잘 맞는 h식을 찾는 것이며 결국 아래 오차식 $J(\theta)$을 최소화 시키는 parameter인 ${\theta}_0, {\theta}_1$를 찾는 것으로 정리할 수 있다.

$$ J({\theta}_0, {\theta}_1) = {1 \over{2m}} \sum_{i=1}^m (h_{\theta}(x^i)-y^i)^2  \quad \quad (2) $$

최종 목표인 오차함수 최소화는 아래처럼 표현한다.

$$\underset{\theta_0, \theta_1}{min} \ J(\theta_0, \theta_1)  \quad \quad (3) $$

위 최소화 문제를 만족시키는 Parameter인 ${\theta_0, \theta_1}$를 구하는 방식 중에 하나가 여기서 얘기하려는 Gradient Descent 방식이며 일종의 Parameter Update Rule이라고 할 수 있다.

결론부터 말하자면 Parameter Update Rule은 아래와 같다.

$$ \theta_i  := \theta_i - \alpha {\partial \over \partial {\theta_i}} J(\theta_0, \theta_1)  \quad \quad (4) $$

$\alpha$는 learning rate라 불리는 임의의 양수이다. Gradient Descent 알고리즘의 동작은 다음의 단계를 거친다.

1. 초기 ${\theta_0, \theta_1}$를 가정한다.
2. Parameter Update rule에 따라 반복적으로 ${\theta_0, \theta_1}$를 변화시킨다.
3. Convergence가 확인되면 최종 ${\theta_0, \theta_1}$를 반환하고 반복을 종료한다.

## Derivation

Gradient Descent의 핵심인 Parameter Upadate 식(4)는 수학적으로 아래와 같이 유도된다.

Cost Function $J({\theta}_0, {\theta}_1)$에 대한 미분은 아래와 같이 근사시킬 수 있다.

$$\Delta J({\theta}_0, {\theta}_1) \approx {\partial J({\theta}_0, {\theta}_1) \over \partial {\theta_0}} \Delta {\theta_0} + {\partial J({\theta}_0, {\theta}_1) \over \partial {\theta_1}} \Delta {\theta_1}$$

$$ = ({\partial J({\theta}_0, {\theta}_1) \over \partial {\theta_0}}, {\partial J({\theta}_0, {\theta}_1) \over \partial {\theta_1}})\cdot (\Delta {\theta_0}, \Delta {\theta_1})$$

$$ = \nabla J_{\theta_i} \Delta{\theta_i}$$

위 식은 ${\theta}$가 변화하는 방향 ($\Delta{\theta_i}$)에 따라 Cost Function이 어떻게 변하는지($\Delta J({\theta}_0, {\theta}_1)$) 알려준다. 

$J(\theta_0 , \theta_1)$를 최소화하기 위해 $\Delta J(\theta_0 , \theta_1)$가 음수가 되도록 ${\theta}$의 방향을 설정해 주자는 것이 Gradient Descent의 주된 아이디어 이다. 이를 위해 $\Delta \theta_i$를 아래 식과 같이 가정해보자. 

$${\Delta {\theta_i} = -\alpha \nabla J_{\theta_i}}  \quad \quad (5)$$

$\alpha$는 learning rate라 불리는 임의의 양수이다. 이어서 $\Delta J$는 아래식과 같이 주어진다.

$$ \Delta J({\theta}_0, {\theta}_1) \approx \nabla J_{\theta_i} \Delta{\theta_i} $$

$$ = \nabla J_{\theta_i} (-\alpha \nabla J_{\theta_i})$$

$$ = -\alpha \left| \nabla J_{\theta_i} \right|^2 \leqq 0 $$

정리해보면 식(5)를 통해 $\Delta J$가 음수가 되도록 $\theta$를 update해주는 방식으로 Cost Function을 최소화 시켜주는 알고리즘이 Gradient Descent 이다. (좀 더 정확히는 Batch Gradient Descent 이다.)

## Application

필요한 식 정리가 끝났다. 실제로 Gradient Descent 알고리즘을 통해 Linear Regression을 해보자. 할 수 있는 가장 간단한 형태의 Machine Learning이다.

계산에 사용할 Python Module을 import 해준다.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

Toy Dataset으로 iris를 이용할 것이다. (붓꽃 꽃받침의 너비/길이, 꽃잎의 너비/길이를 모아놓은 dataset이다.)

```python
csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
col_names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Class']
data =  pd.read_csv(csv_url, names = col_names)
```

변수들간의 관계를 알아보기 위해 두개씩 선택하여 Plot 해본다.
Septal_length는 꽃받침 길이, Septal_width는 꽃받침 너비, Petal_length는 꽃잎 길이, Petal_width는 꽃잎의 너비이다.


```python
x1 = np.array(data['Sepal_Length'].to_list())
x2 = np.array(data['Sepal_Width'].to_list())
x3 = np.array(data['Petal_Length'].to_list())
x4 = np.array(data['Petal_Width'].to_list())

plt.subplots(constrained_layout=True)
plt.subplot(2, 3, 1), plt.scatter(x1, x2), plt.xlabel('Sepal_Length'), plt.ylabel('Sepal_Width'), plt.title('x1 vs x2')
plt.subplot(2, 3, 2), plt.scatter(x1, x3), plt.xlabel('Sepal_Length'), plt.ylabel('Petal_Length'), plt.title('x1 vs x3')
plt.subplot(2, 3, 3), plt.scatter(x1, x4), plt.xlabel('Sepal_Length'), plt.ylabel('Petal_Width'), plt.title('x1 vs x4')
plt.subplot(2, 3, 4), plt.scatter(x2, x3), plt.xlabel('Sepal_Width'), plt.ylabel('Petal_Length'), plt.title('x2 vs x3')
plt.subplot(2, 3, 5), plt.scatter(x2, x4), plt.xlabel('Sepal_Width'), plt.ylabel('Petal_Width'), plt.title('x2 vs x4')
plt.subplot(2, 3, 6), plt.scatter(x3, x4), plt.xlabel('Petal_Length'), plt.ylabel('Petal_Width'), plt.title('x3 vs x4')

plt.show()
```

![Output](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-08/output_4_0.png)

오른쪽 아래 그림(꽃잎 길이 - 꽃잎 너비)가 가장 선형으로 보인다. 이 Dataset만 쓰기로 한다.

```python
x = x3 # Petal Length
y = x4 # Petal Width
```

선형식인 $h = \theta_1 x + \theta_0$와 Cost Function $J(\theta_0, \theta_1)$을 함수로 선언한다. (각각 y_pred, J_tot로 이름붙였다.) 편의상 기울기 $\theta_1$을 m으로, 절편 $\theta_0$를 b로 표현한다. 

m=0.4, b=-0.5일때 얼추 잘 맞는 것으로 보이며 이 때 Cost값은 0.04 정도이다.

```python
def y_pred(m, b, x):
    return m*x+b

def J_tot(m, b, x, y):
    return (1/(2*len(x)))*sum(((m*x+b)-y)**2)

plt.plot(x, y_pred(0.4,-0.5,x),'r')
plt.scatter(x, y)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

print(J_tot(0.4, -0.5, x, y))
```
    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-08/output_6_0.png)   


    0.040613333333333335


가장 핵심인 Gradient Descent 알고리즘 부분이다. learning rate $\alpha$는 0.1로 했으며 parameter m, b의 초기값은 각각 1.5, -4로 시작하였다. 1000번의 반복문을 돌게 하였으며 이전 계산에 비해 Cost가 0.00001도 감소하지 않는다면 수렴했다고 가정하고 반복문을 나오도록 한다. 총 242번 계산끝에 수렴되었다.

```python
m = 1.5
b= -4

J_list = []
m_list = []
b_list = []

alpha = 0.1
iteration = 1000
tol = 0.00001

for i in range(iteration):
    m_list.append(m)
    b_list.append(b)

    grad_m = (-1/len(x))*np.sum((y-(m*x+b))*x)
    grad_b = (-1/len(x))*np.sum(y-(m*x+b))

    m = m - alpha*grad_m
    b = b - alpha*grad_b

    J_list.append(J_tot(m, b, x, y))
    if i>2:
        if J_list[i-1]-J_list[i] < tol:
            print(f"{i}번째 계산 끝에 수렴 완료")
            print(f"기울기 = {m}, 절편 = {b}")
            break

```

    242번째 계산 끝에 수렴 완료
    기울기 = 0.4286328132289579, 절편 = -0.4219131892313386
    

아래코드는 parameter m, b에 따른 Cost값 Contour map으로 표현하고 학습이 진행되는 동안 parameter update되는 궤적을 표현해준다.

```python
xlist = np.linspace(-1, 2, 400)
ylist = np.linspace(-5, 5, 400)

X, Y = np.meshgrid(xlist, ylist)
Z = np.zeros(X.shape)
```


```python
for i in range(X.shape[0]):
    for j in range(X.shape[0]):
        Z[i][j] = J_tot(X[i][j], Y[i][j], x, y)
```


```python
m_list_p = [m_list[i] for i in range(len(m_list)) if i % 5 == 0]
b_list_p = [b_list[i] for i in range(len(b_list)) if i % 5 == 0]

fig, ax = plt.subplots(1, 1)
levels1 = np.linspace(0.03, 0.1, 2)
levels2 = np.linspace(0.2, 1, 3)
levels3 = np.linspace(4, 30, 4)

levels = np.concatenate([levels1, levels2, levels3])
cp = ax.contour(X, Y, Z, levels)
plt.plot(m_list_p, b_list_p, 'rx-')
plt.title("Contour map for Cost with m and b")
plt.xlabel("m")
plt.ylabel("b")
plt.show()
```
![output](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-08/output_10_0.png)

빨간선이 parameter update되는 궤적을 나타낸다. Cost가 가장 작은 부분인 가운데 작은 등고선으로 parameter가 이동하는 것을 볼 수 있다.

최종 결과를 Plot 해본다. 반복계산 횟수(학습 횟수라고도 부른다)에 따라 Cost가 감소하는 것을 알 수 있으며 최종 결과식과 data의 경향성이 잘 맞는 것을 확인할 수 있다.

```python
plt.plot(J_list)
plt.title("Cost vs No. of Iteration")
plt.xlabel("No. of Iteration")
plt.ylabel("Cost")
plt.show()

plt.title("Calculated Results")
plt.plot(x, y_pred(m, b, x))
plt.scatter(x, y)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()
```


    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-08/output_11_0.png)
    
    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-08/output_11_1.png)
    

## Summary

Labeled Data(붓꽃 꽃잎의 길이-너비 data)를 이용해서 첫번째 **Machine Learning**을 해보았다. Cost Function을 최소화하는 선형식 h의 parameter를 Gradient Descent 알고리즘으로 구해보았고 실제로 잘 작동하는 것을 확인했다. 모든 경우에 잘 작동하는 것은 아니며 Cost Function이 Convex한 형태여야만 global minimum에 도달할 수 있으며 아닌 경우 local minimum에 빠질 수 있다. 더 궁금해서 자세한 설명이 필요하면 [이 문서](http://sanghyukchun.github.io/63/)혹은 Andrew Ng교수의 수업을 추천한다.

앞에서 얘기했던 Machine Learning의 정의를 생각해 보면 (Data로 부터 학습하는 알고리즘) '학습'이라는 단어에 대해 이제는 다른 느낌을 가질 수 있다. 처음에는 마치 컴퓨터가 인간처럼 학습을 한다는 뭔가 공상과학속 얘기같은 느낌을 받았을 수 있다. 하지만 실상은 Gradient Descent 알고리즘이 제안하는 parameter update 방식을 반복해서 적용하는 것(오차를 최소화 해나가는 과정)임을 알 수 있다. 결국 Cost를 최소화 해나가는 반복된 parameter update과정이며 본질적으로 Fitting이고 Regression이다. (내가 잘 몰라서 이렇게 얘기하는 것일수도 있지만 어쨌든 지금까지 내 생각은 이렇다.)

물론 본 예제에서 수행한 Linear Regression은 굳이 이런 Gradient Descent를 쓰지 않더라고 코드 한줄로 답이 나오게 하는 함수들이 많다. 하지만 이 알고리즘을 이해하는데는 분명히 도움이 되었으리라 생각한다. 다음 Posting에서 인공신경망으로 학습을 시킬 때 다시한번 Gradient Descent를 사용할 것이다.