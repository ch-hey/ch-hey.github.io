---
title: "[머신러닝] 공부 시작한다면 이 순서로"
date: 2023-01-04 19:55:00 +0800
categories: [Machine Learning, Syllabus]
tags: [Coursera, Andrew NG, AI, Machine Learning]
math: true
mermaid: true
pin: true
---

![Machine Learning Coursera](https://miro.medium.com/max/1100/1*FGGge_GilZ_KJYaoryaxkA.webp){:. width="600"}

매우 건방진 얘기 같겠지만 체계적으로 머신러닝 기초 공부를 시작 해보고 싶다면 아래 순서대로 해보시기를 추천드린다.

- What is Machine Learning?
- Linear Regression
- Gradient Descent
- Logistic Regression
- Artifical Neural Network
- Decision Tree
- Supervised Machine Learning Applications
- Some Unsupervised Machine Learning (Clustering, Anomaly detection, PCA)

처음 머신러닝이란것에 관심이 생겨서 공부해보려 했을 때 인터넷에 수많은 정보 중에 기초부터 다질만한 체계적인 과정을 접하기 까지 많은 시간을 헤맸었다. 다른 포스팅에서도 종종 말했었던 Coursera 강의내 목차들을 살펴보면서 공부 순서를 한 번 정리해본다.

## 1. Intro

[Coursera](https://www.coursera.org/)라는 강의 플랫폼이 있다. 제한적이지만 일정기간 무료로 강의를 들을 수 있다.

여기서 들었던 Andrew NG 교수의 강의 목차를 소개하려고 한다. 둘 다 들으면 좋고 아니면 이 흐름대로 공부해보자. 개인적으로 머신러닝 기초를 다지고 어느정도의 기능구현을 시작해보는데 이보다 좋은 방법은 없을 것 같다. 

- [Supervised Machine Learning: Regression and Classification](https://www.coursera.org/learn/machine-learning)

이 강의는 [GNU Octave](https://octave.org/)라는 무료 수치해석 프로그램으로 실습해나가면서 머신러닝 공부하는 강의다. (Matlab의 무료버전으로 생각하면 된다. 실제로 m파일 수정하고 실행한다.) 내 머신러닝 첫 강의였고 개인적으로 굉장히 만족스러웠었다. 머신러닝이 뭔지, Regression이 뭔지 이런거 수학으로 얘기하고 실습한다. 

매주 코드짜는 과제 수행해야한다. 가능하면 직접 해보되 답은 구글링하면 있으니 걱정하지 말고 시간이 없다면 배껴서 제출하자. 

- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?utm_campaign=WebsiteCourses-MLS-TopButton-mls-launch-2022&utm_medium=institutions&utm_source=deeplearning-ai)

이 강의를 좀 더 추천드린다. 꽤 최근에 마음먹고 들었는데, 이 강의는 Python 기반으로 Tensorflow와 Scikit-learn같은 머신러닝 모듈도 경험해 볼 수 있다. 7일간 무료인데 주말끼고 해도 다 듣느라 일주일내 다 듣는게 정말 힘들었다. 위 강의를 먼저 들었던 덕에 그나마 가능했던 일정이였다고 생각한다. 7일 후에는 $45/월 정도 지불해야 들을 수 있고 여유가 된다면 돈내고 차근차근 듣는것도 나쁘지 않다고 본다. 

마찬가지로 매주 코드짜서 제출해야 하는 과제가 있다. Github에 누군가 답을 올려놨으니 시간없다면 [여기](https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera) 한번 보자.

이번 포스팅에서는 이 강의에서 가르쳐주는 내용과 목차 간단히 정리해 보려고 한다.

## 2. Coursera - Machine Learning Specialization

이 강의는 아래 3개 챕터로 이루어져 있다.

- Supervised Machine Learning: Regression and Classification
- Advanced Learning Algorithms
- Unsupervised Learning, Recommenders, Reinforcement Learning

뭘 배우는지 하나씩 살펴보자.

### 2.1. Supervised Machine Learning: Regression and Classification

이 챕터 하나가 3주짜리 강의로 구성되어 있다. 7일안에 3개 챕터 다 배워야 하니 실상은 한 주 강의를 하루안에 무조건 돌파해야 한다. 어찌됬든 각 주마다 배우는 내용은 아래와 같다.

- Week 1: Introduction to Machine Learning
  - Overview of Machine Learning
  - Supervised vs. Unsupervised Machine Learning
  - Regression Model
  - Train the model with gradient descent

- [x] Keyword: Jupyter Notebook, Python, Linear Regression, Cost function, Gradient descent

첫 주에는 머신러닝 Intro다. 머신러닝이 뭔지, 지도-비지도학습이 뭔지, Regression이 뭔지, 이때 쓰는 Gradient descent 알고리즘에 대해 설명한다. 관련 내용들은 이 블로그에도 나름 정리해둔 포스팅들이 있지만 강의를 듣는게 더 좋다. 

Python으로 실습이 수행되고 Jupyter Notebook으로 한다. 따로 설치할 필요 없이 웹에서 실행가능하다.

- Week 2: Regression with multiple input variables
  - Multiple linear regression
  - Gradient descent in practice

- [x] Keyword: Vectorization, Numpy, Multiple Linear Regression, Feature Scailing, Convergence, Learning rate, Feature Engineering

두번째 주에는 Regression을 자세하게 실습해본다. 위 키워드들에 대해 공부하는데, 이런 걸 알아둬야 한다는 느낌으로 봐도 될 것 같다.

- Week 3: Classification
  - Classification with logistic regression
  - Cost function for logistic regression
  - Gradient descent for logistic regression
  - The problem of overfitting

- [x] Keyword: Logistic regression, Decision Boundary, Cost function, Scikit-learn, Overfitting, Regularization

세번째 주에는 Logistic regression으로 분류에 대해 배운다. Lienar regression 공부할 때와 비슷한 느낌이지만 Cost function이 다르다. Entropy라는 단어도 볼 수 있다. 학문들간에 연결되어 있다는 느낌이 드는 부분도 꽤 재밌다.

Linear/Logistic regression 이 두 개는 자연스럽게 뒤에 배울 인공신경망으로 이어지는데 중요한 역할을 한다. 

### 2.2. Advanced Learning Algorithms

이 챕터는 4주짜리다. 쉴생각은 꿈도꾸지말고 달리자.

- Week 1: Neural Networks
  - Neural Networks Intuition
  - Neural network model
  - Tensorflow implementation
  - Neural network implementation in Python
  - Speculations on artificial general intelligence (AGI)

- [x] Keyword: Neural network, Neuron and Layer, Tensorflow, Matrix multiplication

첫번째 주는 드디어 인공신경망에 대해 배운다. Tensorflow로 간단한 실습도 해본다. Numpy로 행렬연산 연습도 하는데 알아두면 무조건 좋다. 앞에서 배운 내용들과 자연스럽게 이어지니 이전 과정들을 잘 들었다면 따라가는데 문제 없을것 같다.

- Week 2: Neural network training
  - Neural Network Training
  - Activation Functions
  - Multiclass Classification
  - Additional Neural Network Concepts
  - Back Propagation

- [x] Keyword: Training, Acitivation function, ReLU, Softmax, Advanced optimization, Back propagation

두번째 주는 인공신경망을 학습시키는 것에 대해 배운다. Multiclass 분류에 대해서도 알려준다.

- Week 3: Advice for applying machine learning
  - Advice for applying machine learning
  - Bias and variance
  - Machine learning development process
  - Skewed datasets

- [x] Keyword: Model evaluation, Model selection, Training/Cross validation/Test data set, Diagnosing bias/variance, Learning curve, Transfer learning, Error matrix, Precision/Recall

세번째 주는 인공신경망 학습중에 주의해야 할 점, 모델을 선택하는 기준같이 실제 머신러닝을 해볼 때 주의해야 할 Practical한 내용에 대해 실습한다. [Tensorflow 블로그에 있는 Regression 예제](https://www.tensorflow.org/tutorials/keras/regression?hl=ko)와 같이 봐도 좋을 것 같다.

추가로 Data가 한쪽으로 쏠려있는 Skewed dataset을 사용하는 경우에 대해서도 알려준다. 뒤에 비정상을 진단하는 Anomaly detection에서도 쓰이는 개념이다.

- Week 4: Decision trees
  - Decision trees
  - Decision tree learning
  - Tree ensembles

- [x] Keyword: Decision tree, Purity, One-hot encoding, Random forest, XGBoost

마지막주는 Decision tree에 대해 배운다. 개인적으로 Regression에서 인공신경망이 가장 좋은 모델이라고 막연히 생각해왔었는데, 이 강의 들으면서 생각이 많이 바꼈다. 좀 더 자세하게 다뤄보고 포스팅 해볼 생각이다.

### 2.3. Unsupervised Learning, Recommenders, Reinforcement Learning

마지막은 3주짜리다. 시간에 쫓기기도 했고 비지도학습은 개인적으로 너무 어렵고 관심도 덜해서 몇개 빼고는 대충 듣긴했다. 그냥 이런게 있구나 하는 정도로 가볍게 들어도 크게 상관 없지 않을까 생각한다.

- Week 1: Unsupervised learning
  - Clustering
  - Anomaly detection

- [x] Keyword: K-means algorithm, Gaussian distribution

첫 주는 비지도 학습 강의에서 그나마 알아듣기 쉬운 부분이었다. 특히 Anomaly detection은 한 번 써먹어볼수도 있겠다는 생각이 들기도 했다. 통계적으로 평균값에 멀어지는 것들을 비정상으로 처리하려는 느낌이었다. 실제로 잘 들어 맞을지는 모르겠지만.

- Week 2: Recommender systems
  - Collaborative filtering
  - Recommender systems implementation detail
  - Content-based filtering
  - Principal Component Analysis

- [x] Keyword: Recommender

두 번째 주는 추천알고리즘에 대해 다룬다. Netflix나 유튜브같은 플랫폼에서 사용자들에게 영상추천해주는 그런것들이다. 마지막에 PCA에 대해서도 말하는데, 언젠가 자세하게 공부해봐야 겠다.

- Week 3: Reinforcement learning
  - Reinforcement learning introduction
  - State-action value function
  - Continuous state spaces

- [x] Keyword: Reinforcement learning, Return, Bellman Equation

비지도학습 끝판왕 강화학습이다. 알고리즘을 어떻게든 설명해 주려고 하지만 잘 이해가 안됬다. 달착륙선이 안전하게 착륙할 수 있도록 학습하는 예제를 하는데 한 번 보고는 전혀 모르겠다. 이해는 안가지만 어떻게든 코드돌려서 학습한 머신러닝 모델로 달착륙선이 넘어지지 않고 잘 착륙하는 움짤 보면 신기하긴 하다.

## Summary

머신러닝을 한다고 해도 이미지나 영상처리에 더 관심이 있다거나 자연어 처리로 챗봇 같은 것에 관심이 있다거나 하는 식으로 관심분야들은 각자 다를 수 있다. 개인적으로는 수식 모델링에 관심이 있어서 그동안 인공신경망과 딥러닝으로 Regression하는 데에만 관심이 있었다. 최근에는 Decision tree나 Anomaly detection도 해보면 재밌을 것 같다. 

관심이 어디있든 위 내용들은 머신러닝 전반에 대한 기초지식이기 때문에 공부해두면 반드시 언젠가 써먹을 일이 생길 것이라고 생각한다.

관련해서 블로그에 몇 개 주제는 이미 포스팅을 했고, 나머지 주제들도 조금씩 써서 나름 정리해 나가려고 한다. 배경 이론, 수식들과 코드로 기능구현까지 각 세부 주제별로 정리해 볼 생각이다. 목차 정리된 이 글에 하나씩 링크 걸어두면 좋을 것도 같다. 기본 뼈대는 Coursera강의 내용이겠지만 너무 베끼지는 말고 가능하면 좀 더 공부도 해서 예제도 바꿔보고 내용도 더 추가해보려 한다.