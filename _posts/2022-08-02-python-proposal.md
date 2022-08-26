---
title: "[Python] 파이썬 하자고 꼬시는 글"
date: 2022-08-02 11:33:00 +0800
categories: [Orientation, Python 하자고 꼬시는 글]
tags: [Python]
math: true
---

Python으로 하는 프로그래밍은 처음에는 정말 막막하고 하나도 모르겠고 어려웠다. 시간이 지나면서 조금씩 경험이 쌓이고 이제는 좀 생각대로 돌아가는걸 확인하니 취미로 이만한것도 없다. 물론 지금도 쉽지는 않다.

화공업계에서 일하면서 공학용 계산에도 써보고 머신러닝도 조금 해보고 나름 자동화도 해보며 주변에 조금씩 티를 내본다. 친한분들에게 해보라고 꼬셔보는데 그렇게 잘 먹히진 않는다. 하지만 나같아도 안했을거 같기 때문에 할말은 없다.

그래서 이번에는 내 얘기 말고 인터넷으로 믿을만한 자료를 검색해본다. Chemical Engineering 이쪽 업계에서 Python은 쓸만한가? 아니 그냥 어디서든 쓸만한가? 추천해도 되나? 언젠간 인기 식어서 아무도 안쓰고 사라지면 어쩌나? 나도 좀 궁금 했었다.

## 1. Is Python Useful?

일단 갓구글에 `is python useful?`이라고 검색해본다. 

2021년 작성된 ['What is Python used for? 10 practical Python uses'](https://www.futurelearn.com/info/blog/what-is-python-used-for)라는 포스팅이다.

> According to the TIOBE index, which measures the popularity of programming languages, Python is the third most popular programming language in the world, behind only Java and C. There are many reasons for the ubiquity of Python, including: 
>> Its **ease of use**. For those who are new to coding and programming, Python can be an excellent first step. It’s relatively easy to learn, making it a great way to start building your programming knowledge.
>> Its **simple syntax**. Python is relatively easy to read and understand, as its syntax is more like English. Its straightforward layout means that you can work out what each line of code is doing. 
>> Its **thriving community**. As it’s an open-source language, anyone can use Python to code. What’s more, there is a community that supports and develops the ecosystem, adding their own contributions and libraries. 
>> Its **versatility**. As we’ll explore in more detail, there are many uses for Python. Whether you’re interested in data visualisation, artificial intelligence or web development, you can find a use for the language. 

쉽고, 문법도 간단하고, 커뮤니티가 많고 쓸데도 많다고 한다.

> So, we know why Python is so popular at the moment, but why should you learn how to use it? Aside from the ease of use and versatility mentioned above, there are several good reasons to learn Python: 
>> Python developers are in demand. Across a wide range of fields, **there is a demand for those with Python skills.** If you’re looking to start or change your career, it could be a vital skill to help you. 
>> It could **lead to a well-paid career**. Data suggests that the median annual salary for those with Python skills is around £65,000 in the UK. 
>> There will be many job opportunities. Given that Python **can be used in many emerging technologies**, such as AI, machine learning, and data analytics, it’s likely that it’s a future-proof skill. *Learning Python now could benefit you across your career*. 

Python 기술이 있는 사람들에 대한 수요가 많다고 한다. 그리고 돈도 많이 준다고 한다!! Python 좀만 더 열심히 하면 월급 좀 오르려나.

아무래도 인터넷에서 프로그래밍 언어를 검색한다는 건 확률적으로 개발자들이 많은 곳을 갈 수밖에 없다. 이들의 기준에서 얘기하는 경우가 꽤 많다고 느껴졌다. 그리고 좋은얘기만 있는 건 아니다. 느리다, 인기는 거품이고 곧 다른언어로 대체될거다 여러 부정적 의견들도 조금씩은 존재한다. 

## 2. Is Python Useful for Chemical Engineering?

아무래도 화공업계에 있다보니 궁금해진다. 이 바닥에서도 좀 쓰려나. 아무도 안쓰고있나? 괜히 불안하다. 구글에 검색해보면 ['Step into the Digital Age with Python'](https://www.aiche.org/resources/publications/cep/2021/september/step-digital-age-python) 이라는 글이 나온다. 2021년 미국화학공학회 AIChE어딘가에 실린 글이다.

### 2.1. Python is already widely used by chemical engineers

첫 section 제목이었다. 이거 이업계에서 꽤 사용되고 있나보다. 안도했다.

> It is widely used by major technology companies across many scientific disciplines, and it benefits from the contributions of researchers. Python is considered to be an **easy-to-learn programming language**, and it is among the first programming languages that many students are taught today. Because more and more students are learning Python, the next generation of engineers will likely be interested in applying it to their professional work. For those out of school, Python is a great programming language to learn independently, with abundant online resources. This time spent **learning the language is a smart investment** for your career. For example, in the Presidential Lecture given at the 2019 AIChE Annual Meeting (Nov. 10–15, 2019, Orlando, FL), Matt Sigelman, CEO of Burning Glass Technologies, mentioned Python specifically as a skill that adds a premium to a worker’s expected salary.

언제나 빠지지 않는 배우기 쉽다는 말 여기도 있다. 어딘가의 CEO가 Python 아는 직원의 기대연봉이 높을거란 얘기도 했단다. 아직 나는 배움이 부족한듯 싶다.

### 2.2. Python adapts to our work

화공업계쪽에도 Python은 적응하고 있다고 한다.

> Python’s greatest power is in its flexibility, and without packages, it would not have its breadth of applications. [Table 1](https://www.aiche.org/resources/publications/cep/2021/september/step-digital-age-python#table%201) highlights some of the most popular enabling packages engineers use to collect and analyze data, perform calculations, and automate tasks.

Python 자체는 open-source이고 사용자도 많기 때문에 수많은 라이브러리 (위에서는 package로 표현)들이 있고 쉽게 사용할 수 있다. 데이터 분석이나 공학용 계산을 하거나 자동화를 수행하는데 쓰인다고 한다. 잘 쓰고 있었나보다.

### 2.3. Python has technical computing capabilities

> When using Python to **solve technical problems**, you may be looking for capabilities from classic numerical methods, or you may be interested in applying a promising new approach (e.g., machine learning) to problem-solving. For either scenario, Python is quite capable.

공학용 계산에서 전통적으로 사용되던 numerical methods에서 채신기술인 machine learning까지 모두 커버할 수 있는 라이브러리들을 갖고있다. 자세한건 직접 해보면 알 수 있다.

## 3. Next Steps

> The aim of this article is to raise awareness of what the Python programming language can do for chemical engineers. Python is capable enough for professional programmers yet simple enough to be taught as an entry-level language. If you are looking to upgrade your skills, consider adding Python to your toolbox. It is free to use, free to learn, and can be used for nearly any digital task.

이 문단으로 하고 싶은 말 끝났다. 화공업계에 있다면 Python 한 번 써보셨으면 좋겠다. 아니어도 한 번 써보면 좋겠다. 입문용 프로그래밍 언어로도 좋다고 한다. 일단 공짜고 배우는 것도 공짜고 생각보다 많은 일을 할 수 있다. 시간만 좀 들이면 된다.

## Summary

긴말 필요없이 한 번 해보자. Python 좋다고 해보시라고 추천할 수 있다. 내 주변에서도 많이들 써서 같이 헤매봤으면 좋겠다. 일단 회사 PC에 Python 설치하고 개발환경도 세팅해보자.