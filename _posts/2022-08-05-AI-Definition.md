---
title: What is Artificial Intelligence?
date: 2022-08-05 20:55:00 +0800
categories: [Machine Learning, Definition]
tags: [AI, Machine Learning]
math: true
mermaid: true
image:
  path: /img/2022-08-06/AI_Character.png
  width: 250
  alt: AI Character
---
무언가에 대해 대화하기 전에 그것에 대한 정의를 서로 맞춰 나가는 것은 매우 중요하다. 시간이 좀 걸리더라도 용어에 대한 정의에 합의한 이후 더 깊게 들어 가는것이 결국 시간을 아끼는 방법이다.

인공지능 (Artificial Intelligence, AI), 머신러닝 (Machine Learing, ML), 딥러닝 (Deep Learning, DL)의 정의에 대해 찾아본 내용 정리다. 여기에는 용어별 정의와 적용/활용 되는 범위와 공부하는데 좋았던 사이트, 자료들 출처정도 정리한다. 다른 포스팅에 간단한 수학과 `Python` 프로그래밍 예제 정리할 계획이다.


## 1. Artificial Intelligence

### Definition

Russell의 ["Artificial Intelligence: A Modern Approach"](http://aima.cs.berkeley.edu/)라는 책에서 AI에 대해 다음과 같이 말한다.

> Historically, researchers have pursued several different versions of AI. Some have defined intelligence in terms of fidelity to human performance, while others prefer an abstract, formal definition of
intelligence called rationality—loosely speaking, doing the “right thing.” The subject matter itself also varies: some consider intelligence to be a property of internal thought processes and
reasoning, while others focus on intelligent behavior, an external characterization. 

AI발전 과정에서 '인간의 행동'에 집중한 연구자들도 있었고 아예 추상적인 '이성'이라는 것에 집중한 연구가 있었던 것 같다. 좀 더 이해하기 쉽고 간단한 정의가 필요해 보인다. [DataRobot](https://www.datarobot.com/wiki/artificial-intelligence/)이란 곳에서 아래와 같이 말한다.

> “**An AI is a computer system that is able to perform tasks that ordinarily require human intelligence**. These artificial intelligence systems are powered by machine learning. Many of them are powered by machine learning, some of them are powered by specifically deep learning, some of them are powered by very boring things like just rules.”

좀 애매한 부분이 있어도 훨씬 간단하고 직관적이다. 인간이 해결할만한 지능이 필요한 문제를 컴퓨터가 해결해주는 것을 AI라고 부르자 하고 말한다. 실은 대부분의 이쪽분야 분들은 위처럼 얘기하는 것 같았다. [여기](https://www.kdnuggets.com/2017/07/rapidminer-ai-machine-learning-deep-learning.html)에서도 AI는 사람의 행동을 흉내내는 모든 컴퓨터 기술을 말한다고 한다. 

그 유명한 Andrew NG 교수님의 [Coursera 강의](https://www.coursera.org/learn/ai-for-everyone?)에서는 AI를 적용 범위에 따라 두 개로 분류하기도 한다. 

AGI (Artificial General Intelligence)
: Do Anything a human can do

ANI (Artificial Narrow Intelligence)
: E.g., smart speaker, self-driving car, web search, AI in farming and factories

AGI는 그냥 사람복사본이고 ANI는 하나의 기능만 가지는 것으로 말하는 것 같다. 

여기까지 정리해보면 

> **AI**는 **`인간이 할만한 지능이 필요한 일을 컴퓨터가 하는 것`** 이라고 정의하자. 
{: .prompt-info}


### Hierachy

구글링해보면 아래 그림같은 다이어그램 많이 볼 수 있다. 

![AI](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-06/AI_ML_DL_Diagram.jpg)

여기서는 AI는 사람의 행동을 흉내내는 기술로 설명한다. AI가 가장 넓은 범주이고 그 안에 머신러닝과 딥러닝이 보인다.

## 2. Machine Learning

AI분야의 처음은 인간이 할 만한 복잡한 일을 컴퓨터가 해결하도록 하기 위해 많은 규칙을 집어넣는 과정이었던 것 같다. 즉 사람이 직접 모든 행동 규칙들을 코드로 짜고 컴퓨터는 주어진 코드대로 행동하는 것이다. 하지만 사진에서 글자를 인식하거나 고양이와 개를 구분하거나 하는 일에 대한 규칙을 정하는 것은 매우 어렵고 어쩌면 불가능하다. 

이와 반대로 수많은 데이터를 입력해주고 컴퓨터가 직접 그 규칙을 찾도록 하는 방식이 제안되었고 보통 이런 방식을 머신러닝이라고 부른다.

### Definition

["Python Programming And Numerical Methods"](https://pythonnumericalmethods.berkeley.edu/notebooks/chapter25.01-Concept-of-Machine-Learning.html)라는 책에서 머신러닝을 아래와 같이 설명한다.

> Machine learning, as the name suggest, are **a group of algorithms that try to enable the learning capability of the computers, so that they can learn from the data or past experiences**. The idea is that, as a kid, we gain many skills from learning. One example is that we learned how to recognize cats and dogs from a few cases that our parents showed to us. We may just see a few cats and dogs pictures, and next time on the street when we see a cat, even though it may be different from the pictures we saw, we know it is a cat. This ability to learn from the data that presented to us and later can be used to generalize to recognize new data is one of the things we want to teach our computers to do.

과거 경험이나 데이터로부터 배우는 행동을 할 수 있는 알고리즘을 머신러닝이라고 정의한다.

Laurence의 [Coursera 강의](https://www.coursera.org/professional-certificates/tensorflow-in-practice)에서는 이를 조금 틀어서 아래와 같이 설명한다.

![img](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-06/ML.PNG){: width="500"}

즉 전통적인 AI분야에서는 미리 하드코딩된 Rule과 Input을 주면 답이 나오는 방식이었지만 현재의 머신러닝은 Data와 답을 주어주면 이를 위한 방법을 결과로 주는 알고리즘을 말한다.

여기까지 정리해보면 

> **Machine Learning**은 AI의 한 분야로 **`Data로 부터 학습하는(답을 주는 Rule을 찾는) 알고리즘`** 이라고 정의하자. 
{: .prompt-info}

되게 추상적이고 머 대단한게 있어보이고 딱 와닿지 않게 느껴질 수도 있지만, 예시를 한 번 보면 그냥 결국 Fitting 하는거구나 생각이 든다.

### Hierachy

머신러닝에 대한 거의 완벽한 분류가 ["Python Programming And Numerical Methods"](https://pythonnumericalmethods.berkeley.edu/notebooks/chapter25.01-Concept-of-Machine-Learning.html)에 있다.

![ML Hierachy](https://pythonnumericalmethods.berkeley.edu/_images/25.01.01-types-of-ML.jpg){: width="500"}

예를 들어 개와 고양이를 구분하는 머신러닝 모델을 만들기 위해 데이터를 모은다고 상상해보자. 개와 고양이 사진을 수천장 모으는데 각 그림파일의 이름에 각각 개인지 고양이인지 적는다면 (Labeling을 한다면) 이는 labeled data이고 이런 경우가 아니면 unlabeled data이다.

답이 함깨 정리된 data를 학습에 사용하는 경우는 Supervised Learning이라고 부르며 '개 or 고양이' 또는 '참 or 거짓'처럼 이산값이 결과값인 경우는 Classification에 해당하고 내일의 기온을 예측하는 경우와 같이 연속적인 값이 결과값인 경우 Regression이다.

답이 함께 정리되지 않은 data를 학습에 사용하는 경우는 Unsupervised Learning이라고 부르며 그림에서는 크게 Clustering과 Dimensionality reduction으로 구분하지만 실은 이분야는 이런식으로 구분하기는 쉽지 않고 그냥 이런 분야가 있다정도로 생각하면 좋을 것 같다.

**참고로 Andrew NG교수의 위 모든 분야를 커버하는 범위의 [Coursera 강의](https://www.coursera.org/learn/machine-learning)가 있다.**

## 3. Deep Learning

위에 Machine Learning 분류표를 보면 맨 아래 네개의 분류가 있다. 이 중 Classification과 Regression을 보면 Neural Network라는 것이 보인다. Artificial Nerual Network, 인공신경망으로 불리는 이것은 머신러닝 기법 중의 하나로 IBM에서 잘 설명한 [**문서**](https://www.ibm.com/kr-ko/cloud/learn/neural-networks)가 있다. 

![Deel Learning](https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_01_03-DeepNeuralNetwork-WHITEBG.png){: width="500"}

위 그림처럼 인공신경망은 입력층, 은닉층, 출력층 등의 여러 층으로 이루어진 시스템이다. 이들중 중간에 은닉층이 2개이상 층으로 이루어진 복잡하고 거대한 인공신경망을 Deep Neural Network라고 부르며 이를 활용한 머신러닝을 Deep Learning이라고 부른다. 

여기까지 정리해보면 

> **Deel Learning**은 Machine Learning의 한 분야로 **`2개 이상의 은닉계층을 가지는 인공신경망을 활용한 머신러닝기법 중 하나`** 라고 정의하자. 
{: .prompt-info}

컴퓨터 기능 향상과 함께 거대한 인공신경망을 활용한 무거운 계산들이 가능해지면서 이미지인식이나 자연어 처리 등의 분야에서 Deel Learning이 많이 활용되었고 실제로 잘 작동하는 것이 최근의 인공지능에 대한 관심을 설명하는 여러 요인 중에 하나일 것 같다.


## Summary

나름 여기저기서 긁어모은 자료들을 정리하고 엮어보았는데 잘못된 정보가 있을 수도 있고 이쪽 분야에서 충분히 합의되지 않은 내용이 들어가 있을 수도 있다. 그리고 이 분야가 워낙 변화가 많은 곳이다 보니 지금은 맞지만 나중에는 아닌 경우가 있을 수도 있다.

따라서 수학으로 정의된 책이나 강의를 보는 것을 추천드린다. 직접 코드짜보고 머신러닝을 해보면 훨씬 더 체감이 되고 이해하기 쉽다. 여기서는 그냥 개괄적인 내용만 다루고 자세한 내용은 다른 포스팅에 하나씩 풀어갈 예정이다.

특히 [Coursera의 Andrew NG교수 강의](https://www.coursera.org/learn/machine-learning)를 추천드린다. 기초를 다지는데 아주아주 좋은 것 같다. 포스팅할 내용의 주된 부분도 이 수업에서 배운 내용을 정리할 예정이다. 이 이후에 Laurence의 [Coursera 강의](https://www.coursera.org/professional-certificates/tensorflow-in-practice)를 들으면 머신러닝을 위한 Tool 사용방법을 접해볼 수 있다. 

앞으로의 Tool로 Python과 Jupyter, 머신러닝 라이브러리는 Tensorflow를 사용할 것이다. 



