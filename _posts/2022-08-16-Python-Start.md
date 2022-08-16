---
title: Python Install and Dev.Env. Set up
date: 2022-08-16 19:55:00 +0800
categories: [Python, Initial Setting]
tags: [Python, Initial Setting, VSCode]
---

![python logo](https://upload.wikimedia.org/wikipedia/commons/f/f8/Python_logo_and_wordmark.svg){:. width="500}

![vscode logo](https://miro.medium.com/max/480/1*MGcLJS1ZvMFcBA94PXn16Q.png){:. width="300}

나같은 프로그래밍 초심자에게 Python은 설치하고 구동하는 것부터가 진입장벽이다. 일단 설치를 해야 뭐라도 할텐데. 심지어 회사 PC에 설치는 신경쓸것도 많고 일단 안되는게 더 많다. 이런 고민 가진 분을 위한 나름의 해결책 작성한다.

참고로 이 포스팅에 모든 오류들을 하나하나 다 작성할 수 없다. 여기 없는 오류들은 가능한 Google검색을 해보자. 추후에 코드를 짤 때에도 가져야 할 좋은 습관이다. Google에 대부분의 답이 있다.

## Intro

Google에 `Python 설치`라고 검색하면 많이 나온다. 몇개 글을 읽어보면 대강 감이 올 것이다. Python을 설치하고 Code작성을 위한 Editor를 설치하고 적절히 환경설정 해주는 작업들이다. 

하지만 회사에서 설치해서 사용하려면 어떻게 해야할까. IT기업이라면 애초에 IT전문인력일테니 회사의 전반적인 인프라에 대해서도 잘 알겠지만 이런 경우가 아닌 일반 회사에서 업무용이나 스터디용으로 설치해보려고 하면 여러가지를 고민해야 한다.

여러가지 시행착오를 거친 결과 License나 회사 정책상 문제가 없었던 방식을 정리한다. 흔하게 겪는 오류와 이에 대한 해결 방법도 같이 메뉴얼처럼 작성해본다.

> 결론은 Python + VSCode 조합이 2022년 8월 16일 현재 기준으로 회사에서도 사용하는데 전혀 문제가 없는 License이다. 
{: .prompt-info}

잡다한 배경설명이 귀찮을 경우 밑의 License부분은 건너뛰고 바로 Install 부분부터 읽으면 된다.

## License

어떤 것이든 회사 PC에서 '사용'하려 한다면 무조건 상업적 사용으로도 문제가 없는 License인지 확인해야 한다. 심지어 설치할 당시에 무료였여도 설치 이후 주기적으로 확인이 필요할 수 있다.

이런부분을 확인하지 못하고 그냥 무지성 설치를 시도한 경우 설치가 안되면 다행이고 회사 전산관련 팀에서 연락이 오면 그나마 다행이다. 최악에는 License 소유회사로부터 어마어마한 비용이 회사로 청구될 수 있다. (대학원 PC에 Matlab 크랙버젼을 설치해도 몇억짜리 청구서가 날아오는 세상이다.)

보통 내가 PC에 설치하고자 하는 어떠한 것들에 대한 공식 페이지가 존재하고 거기에 License관련 조항들이 있는 경우가 대다수다. 여기서 내가 설치하려고 하는 어떠한 것이 어떤 License를 선언했는지 확인하고 상업적인 용도로도 무료인지 확인하는 절차가 필요하다.

### Python

**결론부터 말하면 Python은 회사에서 '사용'하는데 전혀 문제가 없는 License이다.**

추가로 '사용'이라고 강조했다. Python으로 작성한 코드나 프로그램을 '배포'나 '판매'하려 하는 건 또 다른 얘기다. 애초에 그게 가능한 분이시라면 이 글을 읽지도 않을 것이다.

### IDE

프로그래밍 언어인 Python은 무료다. 다만 Python만 설치한다고 해서 '편하게' Code짤수는 없다. IDE 혹은 개발환경이라고 부르는 일종의 Code Editor가 필요하며 이부분이 여러 선택지와 함깨 여러 License들이 존재한다. 



적당히 Google에 `python 개발환경 추천` 이라고 치면 여러 글들이 있다. [이 문서](https://coding-kindergarten.tistory.com/4) 추천드린다.

먼저 Anaconda, 아나콘다라는게 있다. 이게 원래는 무료였는데 **어느새인가 유료로 변환되었다.** 개인에게는 무료이니 개인 PC에서 처음 시작해볼때는 Anaconda도 추천드린다.

또 다른건 Pycharm이다. 이건 애초에 유료다. 

VSCode라는 것도 보일 것이다. [공식 홈페이지](https://code.visualstudio.com/)내 [License 문서](https://code.visualstudio.com/License)에서 아래와 같이 말한다.

> Source Code for Visual Studio Code is available at https://github.com/Microsoft/vscode **under the MIT license agreement** at https://github.com/microsoft/vscode/blob/main/LICENSE.txt. Additional license information can be found in our FAQ at https://code.visualstudio.com/docs/supporting/faq.

MIT License Google에 검색해보자. 아마 상업적 용도로도 무료라고 설명되어 있을 것이다. 

VSCode는 무려 마이크로소프트에서 무료로 풀어버린 Code Editor고 python외에도 여러가지 프로그래밍 언어를 편집하고 구동할 수 있다. 그리고 여러 Extension들이 있어서 라이센스 확인후 설치만 하면 꽤 있어보이게 코드를 작성할 수 있다. 사용자도 굉장히 많은 편이니 만약에 중간에 라이센스가 변경이 되더라도 생각보다 쉽게 알 수 있을 것이다.

### 회사 정책

'회사 정책상 Python 쓰면 안됩니다' 이런곳은 아마 거의 없을 것이다. 그래도 한 번 회사 전산관련팀에 문의해보자. 물어보면 담당자를 알려주든 아니면 대부분 그냥 쓰세요 할 것이다. 

하지만 중간중간 라이센스 확인은 사용자의 몫인가보다. 인터넷 서핑하다가 Anaconda가 유료로 바뀐걸 알았을 때 황급히 지웠던 경험이 있다. 회사에서는 알려주지 않는다. 당연한 얘기지만 좀 그렇다.

## Install

Intro에서 밝힌대로 Python과 VSCode 설치해본다.

### Python

[점프 투 파이썬](https://wikidocs.net/8) 01-4 파이썬 설치하기 부분을 보면 자세하게 나와있다. 이대로 진행해도 무방하다. 어차피 Python은 회사에서도 무료다.

[공식페이지](https://www.python.org/)에서 Python 설치를 위한 installer 다운받고 설치한다. 특정 version이 필요한 경우도 있다. 드물지만 설치된 Python version에 따라 같은 code라도 정상 작동하기도 하고 작동 안하기도 한다. 나는 현재 3.9.6 version을 사용중이며 같은 version을 설치하려는 경우 [여기](https://www.python.org/downloads/release/python-396/)로 가면 된다. 

### VSCode

[공식페이지](https://code.visualstudio.com/)에 들어가면 바로 Download for Windows뜬다. Version상관없이 그냥 다운받고 설치한다. 

설치 중간에 옵션 선택하는 부분에서 아래와 같이 설정해주면 편하다. 까먹고 안해도 상관없다. 나머지는 모두 yes로 통과한다.

![img](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-16/intstall_1.PNG){:. width = "300"}

Google에 `vscode 설치`라고 검색해도 많이 나온다. VSCode도 마찬가지로 **아직은 무료**라서 큰 걱정 없이 설치하자. (현재는 2022년 8월 16일이다.)

### Extensions

아직 끝이 아니다. 추가 기능인 extension을 설치해야 한다.

VSCode 설치가 완료되어 실행시켜보면 아마 아래 그림처럼 나올 것이다. Extension은 빨간 박스를 클릭하면 검색할 수 있다.

![img](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-16/install_2.png){:. width = "100"}

python을 실행하려면 추가 기능을 설치해야 한다. 아래처럼 빨간 박스에 python검색하면 큰 화면에 설치 가능한 화면이 뜰 것이다. 사진은 이미 설치된 PC에서 캡쳐한 사진이다.

![img](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-16/install_3.png){:. width = "300"}

아래 3개의 extension 반드시 설치해야 한다. 검색해서 설치하자.

1. python: 설치하면 pylance등의 추가 extension들이 자동으로 설치된다.
2. Jupyter: 연구/간단한 업무/코딩 스터디 등에서 처음에 접할 때 굉장히 편한 tool이다. 일단 설치하자.
3. Prettier: Style을 이쁘게 해준다. 보기 좋은게 최고다.

## Hello World!

이제 python code를 작성하고 실행시켜 볼 것이다. Code Editor로 VSCode를 활용해서 Jupyter로 작성/구동해 볼 것이다. 아래 영상대로 진행해보자. (영상은 gif로 반복이 안걸려있다. 처음부터 보려면 새로고침(F5) 눌러본다.)

![gif](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-16/Animation_f.gif){:. width = "300"}

글로 설명하면 아래순서로 진행된다.

VSCode는 폴더를 작업환경으로 열 수 있다. 바탕화면에 python이라는 폴더를 하나 만들어보자. 이후에 VSCode 상단 메뉴에서 File > Open Folder로 방금 만든 python폴더를 선택하고 ok를 누른다.

자동으로 Explorer 탭이 열리면서 python 폴더가 보인다. 폴더명 옆에 파일추가 모양 아이콘을 클릭하면 파일 이름을 입력할 수 있다. Jupyter는 확장자가 `ipynb`인 파일을 읽는다. 첫 파일은 `test.ipynb`로 적고 Enter누르면 메인 화면에 셀이 하나 뜬다. 이 곳에 코드를 작성하면 된다.

메인 화면에 생긴 셀에 아래 코드처럼 작성해본다. 

```python
print("Hello World!")
```

실행은 셀단위로 수행된다. 오른쪽 상단에 마우스를 대면 세모모양 아이콘이 뜰 것이다. 이 아이콘을 클릭한다. (위 영상을 참고한다.) 아마 오류가 뜰 것이다. 이 오류 메시지를 그대로 Google에 검색해보면 해결책이 나오는 경우가 많다. 습관처럼 해보는 것도 좋지만 아래 포스팅대로 진행해본다.

높은확률로 맨 처음 Jupyter 실행할 때 ipykernel이란 python라이브러리를 설치해야 하기 때문에 뜨는 오류다. 라이브러리는 VSCode Extension과는 또 다르다.

terminal을 열고(상단 메뉴바에서 Terminal > New Terminal을 선택하면 된다.) 아래와 같이 써보자. 혹은 검은색 CMD 커맨드창에서 실행해도 된다.

```
pip install ipykernel
```
아래 영상처럼 써보고 Enter 치면 된다. (영상은 gif로 반복이 안걸려있다. 처음부터 보려면 새로고침(F5) 눌러본다.)

![gif](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-16/Animation_f2.gif){:. width = "300"}



아마 회사인터넷을 사용하는 대부분의 경우 또다시 오류가 뜰 것이다. [이 문서](https://melonicedlatte.com/2020/08/12/082300.html)를 참고하자. 읽기 귀찮으면 오류 해결을 위해 아래와 같이 써본다.

```
pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org ipykernel
```
`pip install`과 설치해야할 라이브러리인 `ipykernel`사이에 뭔가 막 작성되 있다. 앞으로 회사 PC에서 Python 라이브러리를 설치할 때 계속 쓰게될 명령어다. 

자 설치가 완료되었다면 다시 셀을 실행해보자. 정상작동 할 것이다.

다시 오류메시지가 뜬다면 미안하지만 Google에 그대로 복사붙여넣기 하고 답을 찾아나서야 한다.

## Summary

정리해보자.

1. Python 설치
2. VSCode 설치
3. VSCode Extension 설치 (Python, Jupyter, Prettier)
4. Terminal에서 ipykernel 라이브러리 설치 (`pip install 설치할 라이브러리`)

여기까지 온 것만 해도 굉장하다. 

이 포스팅에 적혀있지 않지만 나타났던 수많은 오류들을 헤쳐왔을 것이다. 현재 기준으로 License상 문제되지는 않는 조합이라 회사에서 설치하고 사용하는데 일단은 문제가 없을 것이다.

설치할 때 모든 오류들은 Google에 그대로 검색해보고 해결책을 나름 찾아보자. 설치하느라 고생했는데 앞으로 계속 Python을 쓰려고 한다면 이런 습관은 필수다. **웹에 거의 대부분의 답이 있다.**

초보자가 작성한 포스팅으로서 위에 작성한 용어들 중에 정확하게 사용되지 않은 경우들이 있다. 예를 들어 Code Editor와 개발환경, IDE는 분명히 다른 것이다. 아나콘다는 IDE이지만 Jupyter나 VSCode, Spyder같은 것들은 Code Editor다. 하지만 초심자에게는 일단은 그냥 그게 그거다. 지금은 기능구현과 이에 필요한 쉬운 설명에만 집중해본다.