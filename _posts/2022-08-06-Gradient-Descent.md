---
title: Gradient Descent
date: 2022-08-06 20:55:00 +0800
categories: [Machine Learning, Gradient Descent]
tags: [Machine Learning, Gradient Descent]
math: true
mermaid: true
---

## Intro

머신러닝의 구동 방법중 하나인 Gradient Descent, 경사하강법에 대해서 간단히 작성한다. 가능한 Andrew NG 교수의 Notation을 따르려고 한다.

머신러닝 중에서 답이 함께 있는 데이터를 사용하는 Supervised learning에 해당하며 단순 Linear Regression에 대한 예제로 설명한다. Data는 대표적인 toy dataset인 iris를 이용한다. 

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

$$ J({\theta}_0, {\theta}_1) = {1 \over{2m}} \sum_{i=1}^m (h_{\theta}(x^i)-y^i)^2 x \quad \quad (2) $$

최종 목표인 오차함수 최소화는 아래처럼 표현한다.

$$\underset{\theta_0, \theta_1}{min} \ J(\theta_0, \theta_1) x \quad \quad (3) $$

위 최소화 문제를 만족시키는 Parameter인 ${\theta_0, \theta_1}$를 구하는 방식 중에 하나가 여기서 얘기하려는 Gradient Descent 방식이며 일종의 Parameter Update Rule이라고 할 수 있다.

결론부터 말하자면 Parameter Update Rule은 아래와 같다.

$$ \theta_i  := \theta_i - \alpha {\partial \over \partial {\theta_i}} J(\theta_0, \theta_1) x \quad \quad (4) $$

Gradient Descent 알고리즘의 동작은 다음의 단계를 거친다.

1. 초기 ${\theta_0, \theta_1}$를 가정한다.
2. Parameter Update rule에 따라 반복적으로 ${\theta_0, \theta_1}$를 변화시킨다.
3. Convergence가 확인되면 최종 ${\theta_0, \theta_1}$를 반환하고 반복을 종료한다.

## Derivation

Gradient Descent의 핵심인 Parameter Upadate 식(4)는 수학적으로 아래와 같이 유도된다.

Cost Function $J({\theta}_0, {\theta}_1)$에 대한 미분은 아래와 같이 근사시킬 수 있다.

$$\Delta J({\theta}_0, {\theta}_1) \approx {\partial J({\theta}_0, {\theta}_1) \over \partial {\theta_0}} \Delta {{\theta_0}} + {\partial J({\theta}_0, {\theta}_1) \over \partial {\theta_1}} \Delta {{\theta_1}}$$

$$ = ({\partial J({\theta}_0, {\theta}_1) \over \partial {\theta_0}}, {\partial J({\theta}_0, {\theta}_1) \over \partial {\theta_1}})\cdot (\Delta {{\theta_0}}, \Delta {{\theta_1}})$$

$$ = \nabla J_{\theta_i} \Delta{\theta_i}$$

위 식은 ${\theta}$가 변화하는 방향 ($\Delta{\theta_i}$)에 따라 Cost Function이 어떻게 변하는지($\Delta J({\theta}_0, {\theta}_1)$) 알려준다. 

Cost Function이 최소화 되려면 $\Delta J$가 음수가 되도록 ($\Delta{\theta_i}$)를 잡자는 것이 Gradient Descent의 주된 idea이고 아래 식처럼 주어질 수 있다.

$${\Delta {\theta_i} = -\alpha \nabla J_{\theta_i}}  \quad \quad (5)$$

$\alpha$는 learning rate라고 불리는 임의의 양수이다. 

Parameter인 ${\theta_i}$의 변화가 위와 같이 주어질 경우 $\Delta J$는 아래식과 같이 주어진다.

$$ \Delta J({\theta}_0, {\theta}_1) \approx \nabla J_{\theta_i} \Delta{\theta_i} $$

$$ = \nabla J_{\theta_i} (-\alpha \nabla J_{\theta_i})$$

$$ = -\alpha (\nabla J_{\theta_i})^2 < 0 $$

정리해보면 식(5)를 통해 $\Delta J$가 음수가 되도록 $\theta$를 update해주는 방식으로 Cost Function을 최소화 시켜주는 알고리즘이 Gradient Descent 이다. (좀 더 정확히는 Batch Gradient Descent 이다.)

