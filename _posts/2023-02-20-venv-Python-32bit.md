---
title: "[Python] Virtual Env(venv)로 가상환경 만들기"
date: 2023-02-20 18:55:00 +0800
categories: [Python, venv]
tags: [Python, Virtual Environment, VSCode, Python 32bit]
pin: false
---

![Virtual Environments](https://miro.medium.com/v2/resize:fit:720/format:webp/1*wC-mrXuYgarKP8bSK3AivQ.png){:. width="600"}

가상환경 설정을 통해 다양한 개발환경 세팅하는 방법을 정리해본다. 

자세하게는 Python 64bit가 설치된 PC에서 python 32bit 개발환경을 가상환경으로 만드는 방법에 대해 정리해본다. vscode를 사용할 것이다. 구글에 'python venv 가상환경' 이렇게 검색해보면 많이들 나온다.

## 1. Intro

가상환경은 뭐고 왜 필요할까.

여태까지 단순한 공학용 계산이나 아주 기초적인 자동화만 했었다. 이런 상황에서는 기존에 잘 사용하던 Python version에서 벗어나 새로운 개발환경이 필요한 경우는 아직까지는 한 번도 없었다. 

최근에 어떤 이슈가 있어서 python `32bit`를 통해 특정 회사에서 제공하는 API를 이용해야 하는 상황을 겪게 되면서 개발환경 세팅을 새로 해야 할 필요가 생겼다.

처음에는 포맷해야 하나 엄청 난감했었는데 가상환경 설정으로 하나의 PC에서 여러개의 새로운 개발 환경 설정 및 세팅이 가능했고 그렇게 복잡하거나 어렵지는 않았다.

이런 경우 외에도 다른 version의 tensorflow를 써야 하는 상황처럼 특정 버젼의 python 라이브러리를 사용해야 하는 경우에도 가상환경을 통해 쉽게 개발환경 세팅을 할 수 있다.

이번달 포스팅 내용은 딱 가상환경 설정까지여서 대놓고 날먹인 수준이다. 구현중인 프로그램 내용이 정리가 되는대로 다음달 포스팅 작성해 보려 한다. 다음달 포스팅이 힘들어질 것도 같다..

## 2. Virtual Enviroment Setting

Phthon 64bit가 설치되어 있는 상황 하에서 32bit python을 쓸수 있는 가상환경을 만드는 게 목적이다. 먼저 Python 32bit를 설치하자. 64bit때와 동일하게 [공식 사이트](https://www.python.org/)에서 32bit 설치파일을 받고 설치해준다.

설치 할 때 환경변수 체크 '안'하는 점만 주의한다. 32bit가 설치된 Directory도 기억해두자. 일부러 Custom으로 설정하지 않는 이상 보통 `C:\Users\~~\AppData\Local\Programs\Python` 이런 경로에 `python**-32` 이런 식으로 폴더가 생길것이다.

### 2.1. Venv

Python을 처음에 설치하게 될 때 자동으로 같이 설치되는 표준 라이브러리가 있다. 가상환경을 만들어주는 `venv`라는 모듈도 이런 표준 라이브러리 중 하나이다. 가상환경을 만들려고 할 때 python이 설치가 되어 있다면 새로운 모듈 설치가 필요 없다는 뜻이다.

먼저 vscode로 커맨드 창을 열거다. 그 전에 default로 Terminal Profile을 Command Prompt로 설정해준다. 처음에는 powershell로 되어 있을 텐데 CMD로 하는 게 추천된다고 한다. 설정 방법은 [여기](https://hianna.tistory.com/349)를 참고하자.

이제 vscode에서 터미널을 열고 가상환경 '폴더'를 만들 위치로 가보자. 커맨드창에서 특정 경로로 가는 방법은 `cd '원하는 directory` 이다. 잘 모르겠으면 구글링 해보자.

아니면 바탕화면에 test라는 폴더 하나 만들고 우클릭해서 vscode로 열어준다. 그리고 terminal열고 아래 명령어를 입력한다

```
python -m venv venv-test
```

이렇게 하면 커맨드 directory내에 venv-test라는 이름의 폴더가 생길 것이다. venv-test라는 폴더 이름은 아무렇게나 지어도 된다. 여기까지 하면 아래 영상처럼 작동하고 가상환경 만들기 거의 다 끝났다.

![](https://raw.github.com/ch-hey/imgcdn/master/img/2023-02-20/1.gif){:. width="600"}


이제 vent-test라는 가상환경 폴더내에 pyvenv.cfg라는 파일을 메모장으로 연다. (우클릭 > 연결프로그램 > 메모장) 아니면 커맨드에서 venv-test 경로로 간 이후에 아래처럼 명령어를 입력한다.

```
notepad pyvenv.cfg
```

아래 그림처럼 보일 것이다.

![](https://raw.github.com/ch-hey/imgcdn/master/img/2023-02-20/pyvenv-cfg.PNG){:. width="600"}

여기서는 PC에 python 64bit version 3.9.13이 설치되어 있었고, 32bit도 3.9.13으로 설치했다. 32bit python을 설치한 경우 위치는 위 그림 home경로에서 마지막에 폴더이름만 Python39-32였고 이 부분만 수정하고 저장한다. 

각자 상황에 맞게 32bit가 설치되어 있는 경로를 home부분에 적절하게 넣어주면 된다.

이후에 vscode에서 `Crtl+Shift+p`로 명령어창을 열고 python:Select interpreter를 클릭해서 venv-tset:venv라고 표시된 항목을 클릭하면 가상환경이 Activate된다. 이 상태에서 terminal을 열면 Directory에 `(venv-test)`라고 뜨는 것을 볼 수 있다. 이러면 성공이다. 이 상태에서 python이라고 커맨드창에 입력하면 32bit로 뜨는 것을 볼 수 있다.

가상환경을 끄고 싶으면 다시 `Ctrl+Shift+p`로 명령어창을 열고 python:Select interpretre에서 gloabal로 표시되어 있는 기존 python 64bit를 클릭해준다. 확인하려면 다시 terminal열고 python이라고 입력하면 64bit로 뜬다. 가상환경이 필요 없어졌다면 venv-test 폴더를 그냥 지워도 된다.

여기까지 설명한 부분은 아래 영상처럼 작동된다.

![](https://raw.github.com/ch-hey/imgcdn/master/img/2023-02-20/2.gif){:. width="600"}

Vscode가 아닌 까만색 Command Prompt로도 가상환경 만들고 설정하고 다 할 수 있다. 
가상환경 만들 때 생긴 venv-test 폴더 안에 Scripts 디렉토리에 접근해서 activate 커맨드를 입력하면 (activate파일을 실행하면) 된다. 가상환경을 끌 때는 동일한 디렉토리에서 deactivate 커맨드를 입력한다. 더 한 것들도 할 수 있겠지만, Python 코드 실행도 Command Prompt로 할 생각이 아니라면 이 정도에서 멈추자.  

까만색 커맨드 프롬프트에서 명령어 좀 쓰다보면 처음 몇 분은 약간 개발자된 느낌에 뽕이 차오르다가 슬슬 불편해지고 후회된다. 가능한 vscode 쓰자.

## Summary

다양한 version의 모듈을 쓰거나 다른 bit의 python을 써야하는 경우 등 다양한 개발환경이 필요한 경우 가상환경 venv 통해 개발환경 세팅을 할 수 있었다.

처음 Python 설치할 때 Anaconda통해 설치했었는데 version관리나 가상환경 관리가 편하다는 애기를 들었던 거 같다. 그런거 한 번도 안써보고 vscode로 갈아탔는데, 돌고 돌아 가상환경이라니. 평생 쓸일 없을 줄 알았는데, 기왕이면 Anaconda 안쓰고 vscode로도 이런거 편하게(?) 해볼 수 있다는거 구글 검색해고 직접 해본 내용들 정리한다.