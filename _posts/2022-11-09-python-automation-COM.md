---
title: "[Python] Automation - 2. COM Object"
date: 2022-11-09 19:55:00 +0800
categories: [Python, Automation]
tags: [Python, Automation, COM Object, Aspen Plus]
math: true
mermaid: true
---

좀 더 고급진 '자동화'라는 것을 해보자. 마찬가지로 Python으로 하고 window 운영체제 한정이다.

조금만 더 자동화에 대해 파보면 나같은 프로그래밍 초심자는 이해하기 힘든 용어들이 쏟아져 나온다. 대표적으로 API가 그랬고 ProgID 등등 뭐 많았다. 아직 잘 모르겠지만 COM Object도 결국 API중에 하나인건가 싶다. 

다 알아두면 좋지만, 지금 당장은 딱 필요한것에 대해서만 짚고 넘어가자. 관심있으면 구글링도 해보고 자동화해보면서 관련한 error도 겪어보고 그러면 될 것 같다. 

확실히 공학용 계산이나 머신러닝에 비해서 **자동화라는 주제는 개발자의 영역**에 더 가까운 것 같다. 

## 1. Intro

GUI 자동화는 앞 포스팅에서 다루었다.

남은 하나인 COM Object에서 COM은 Component Object Model의 약자로 소프트웨어간 interact를 할 수 있도록 해주는 Microsoft에서 제공하는 어떤 표준 시스템 정도로 생각하면 된다. 나도 잘 몰라서 더 자세히 설명할 자신이 없다. 한 번 코드를 짜서 돌려보면 어떤 느낌인지는 알 수 있다. 

더 자세한 설명은 한국어로 [여기](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=ymy203&logNo=70104910502), 영어로 [여기](https://pbpython.com/windows-com.html)를 보자. 영어로 된 부분을 두 번 추천한다. '이론적으로는' 윈도우 내 모든 프로그램을 python으로 제어할 수 있는 것 같다.

뭔가 모든 프로그램을 제어하기 위한 코드를 짤 수 있는 일반적인 방법이 없을까 찾아본 적이 있었는데, 아무래도 내 수준에서는 아직 어렵다.

## 2. COM Object

### 2.1. Win32com

어찌됬든 이 COM Object를 python으로 사용할 수 있게 해주는 `win32com`이라는 모듈이 있다. GUI자동화에 비해 훨씬 할 수 있는 일들이 많고 잘 짜놓으면 오류도 적지만, 그만큼 알아야 할 것도 많고 어렵다. 좀 더 깊은 레벨의 자동화고 코드뿐만 아니라 제어해야 할 프로그램 자체에 대해서도 자세히 알아야 한다. 잘 안쓰이는 프로그램에 대한 자동화는 구글링해도 정보가 잘 없다.

먼저 [워드](https://m.blog.naver.com/anakt/221874907407), [엑셀](https://wikidocs.net/153820) 또는 [Outlook](https://pbpython.com/windows-com.html)같은 프로그램을 간단하게 제어하는 예제들은 인터넷에 많다. 한 번쯤 따라해보면 감을 갖기 좋을 것 같다. 

### 2.2 Define Automation Task

이번 포스팅에서는 공정시뮬레이션 프로그램인 Aspen Plus V11을 Python으로 제어해본다. 먼저 [이곳](https://kitchingroup.cheme.cmu.edu/blog/tag/aspen/)과 [이곳](https://github.com/YouMayCallMeJesus/AspenPlus-Python-Interface), 그리고 Matlab 커뮤니티에서 많은 힌트를 얻었다. 물론 모든 답은 구글에서 얻었다.

공정시뮬레이션 프로그램 제어를 통해 해보려는 것은 일종의 반복계산, Case Study이다. 잘 되면 최적화에도 써봄직하다. ['Teach Yourself the Basics of Aspen Plus'](https://www.wiley.com/en-us/Teach+Yourself+the+Basics+of+Aspen+Plus,+2nd+Edition-p-9781118980590)라는 책에 있는 예제를 약간 변경해서 해본다.

![컴프레서 두개 열교환기 하나 공정도면](https://raw.github.com/ch-hey/imgcdn/master/img/2022-11-09/process.png)

섭씨 90도, 1bar 짜리 프로판 가스를 2개의 컴프레서를 이용해 최종적으로 8bar까지 가압하려고 한다. 첫 번째 컴프레서에서 배출되는 프로판 가스를 섭씨 90도로 다시 냉각시키는 열교환기가 있으며 열교환기를 거치면서 압력손실은 없다. 두 개의 컴프레서 모두 Polytropic Efficiency = 0.72, Mechanical Efficiency = 1 이다. 

이 때 첫 번째 컴프레서 배출 압력에 따라 두 컴프레서에서 사용하는 전력 합 경향성을 보고 가장 적은 일을 투입하는 경우는 언제인지 확인하려고 한다.

### 2.3 Aspen Plus Control with win32com

코드 짜서 돌려보자. 

```python
import win32com.client as win32
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
```
사용할 모듈은 COM Object 사용을 위한 win32com, 나머지는 다른 포스팅에서도 많이 썼던 모듈들이다.


```python
filename = "TwoCompressors.bkp"

AspenSimulation = win32.gencache.EnsureDispatch("Apwn.Document.37.0")
print("Aspen is Launching...")
AspenSimulation.InitFromArchive2(os.path.abspath(filename))
print(f"{filename} file is opening...")
AspenSimulation.Visible = True
```

    Aspen is Launching...
    TwoCompressors.bkp file is opening...
    
Aspen Plus 파일을 하나 만들었고 이름은 TwoCompressors.bkp라는 파일이다. `EnsureDispatch()`라는 함수 안의 인수로 제어하고자 하는 프로그램, application의 이름이 들어간다. 엑셀이면 'Excel.Application' 이라고 써준다. 

그런데 정확한 이름을 찾는 것이 쉽지가 않다. ProgID라고 하던데, 누군가는 CLSID라고도 하고 어떤사람은 레지스트리에서 찾으라고 하고 어떤사람은 특정 프로그램을 활용하라고도 한다. 

나는 그냥 Aspen Plus 자동화 코드 구글에 돌아다니는 것에서 눈치껏 썼다. 끝에 37은 v11이고 38은 v12 뭐 이런식인 것 같았다. `visible = True`는 프로그램이 실행되는걸 보겠다는 뜻이다. 위까지 실행한 결과는 아래와 같다.

![](https://raw.github.com/ch-hey/imgcdn/master/img/2022-11-09/ani1.gif)


```python
work_compr1 = []
work_compr2 = []

pres = np.linspace(1.1, 7.9, num = 10)

for i in range(len(pres)):
    AspenSimulation.Tree.FindNode("\Data\Blocks\B1\Input\PRES").Value = float(pres[i])
    AspenSimulation.Run2()
    work_compr1.append(AspenSimulation.Tree.FindNode("\Data\Blocks\B1\Output\WNET").Value)
    work_compr2.append(AspenSimulation.Tree.FindNode("\Data\Blocks\B3\Output\WNET").Value)
    print(f"At P_dis = {pres[i]}, total work for compressor = {work_compr1[i] + work_compr2[i]}")

plt.plot(pres, np.array(work_compr1)+np.array(work_compr2))
```

    At P_dis = 1.1, total work for compressor = 6.076464244
    At P_dis = 1.8555555555555556, total work for compressor = 5.80543241
    At P_dis = 2.6111111111111116, total work for compressor = 5.73576098
    At P_dis = 3.366666666666667, total work for compressor = 5.74051231
    At P_dis = 4.122222222222223, total work for compressor = 5.78007698
    At P_dis = 4.877777777777778, total work for compressor = 5.83782127
    At P_dis = 5.633333333333335, total work for compressor = 5.905763298999999
    At P_dis = 6.388888888888889, total work for compressor = 5.979641303999999
    At P_dis = 7.144444444444446, total work for compressor = 6.056999866
    At P_dis = 7.9, total work for compressor = 6.136400717
    
    [<matplotlib.lines.Line2D at 0x1180c6d4700>]

    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-11-09/output_2_2.png)    


첫번째 컴프레서의 배출압력을 바꿔가면서 두 개 컴프레서를 돌리는데 필요한 전력값을 확인한다. `AspenSimulation.Tree.FindNode("\Data\Blocks\B1\Input\PRES")` 이 부분에서 FindNode함수 안의 인수는 첫 번째 Compressor의 이름인 B1의 Input값으로 넣어주게 되어 있는 배출 압력이다. 이 부분을 반복적으로 변경해주면서 전력값을 받아낸다. 이런 일종의 주소, ID를 어떻게 알아내느냐가 대부분의 프로그램 자동화의 어려운 점이다. Aspen Plus에서는 프로그램안에서 확인 할 수 있다. Customize 탭에서 'Variable Explorer'에서 확인이 가능하다. 자세한 내용은 [이 문서](https://chejunkie.com/knowledge-base/navigating-variable-explorer-aspen-plus/)를 보자.


Aspen Plus를 일종의 계산기로 쓰고 Python으로 반복적인 Task를 주는 것으로 이해하면 된다. 그림을 보면 대략 2.5 ~ 3 bar 사이에 최소값이 존재하는 것으로 보인다. 

아래 영상처럼 실행된다. 

![](https://raw.github.com/ch-hey/imgcdn/master/img/2022-11-09/ani2-min.gif)


```python
def net_work(p):
    p = float(p)
    AspenSimulation.Tree.FindNode("\Data\Blocks\B1\Input\PRES").Value = p
    AspenSimulation.Run2()
    w1 = AspenSimulation.Tree.FindNode("\Data\Blocks\B1\Output\WNET").Value
    w2 = AspenSimulation.Tree.FindNode("\Data\Blocks\B3\Output\WNET").Value
    return w1+w2

sol = minimize(net_work, 4)
print(sol)
```

          fun: 5.77215423
     hess_inv: array([[1]])
          jac: array([0.67108864])
      message: 'Desired error not necessarily achieved due to precision loss.'
         nfev: 48
          nit: 0
         njev: 18
       status: 2
      success: False
            x: array([4.])
    

이번에는 아예 함수를 하나 짜서 `scipy.optimize`안에 `minimize`를 써서 최적화 해본다. 초기값은 4로 넣었다. 결과는 수렴하지 않고 'Desired error not necessarily achieved due to precision loss.'라는 오류를 띄우고 초기값 그대로의 값을 return한다. 초기값을 바꿔도 비슷한 결과가 나온다.

구글검색을 해보면 stackoverflow에 [이 문서](https://stackoverflow.com/questions/24767191/scipy-is-not-optimizing-and-returns-desired-error-not-necessarily-achieved-due)가 적당해 보인다.

뭔지는 모르지만 `minimize`함수안에 method를 Nelder-Mead로 지정해 보라고 한다. 아마도 `minimize`함수에서 default로 제공해주는 최적화 방식이 위 문제를 해결하는데 수치해석적으로 적당하지 않았나보다. 자세한건 scipy documentation에서 `minimize`에 대한 것들을 보면 될 것이다. Nelder-Mead방식에 대해서도 알아보면 좋겠지만, 일단 넘기자.

아래와 같이 코드를 수정한다.

```python
def net_work(p):
    p = float(p)
    AspenSimulation.Tree.FindNode("\Data\Blocks\B1\Input\PRES").Value = p
    AspenSimulation.Run2()
    w1 = AspenSimulation.Tree.FindNode("\Data\Blocks\B1\Output\WNET").Value
    w2 = AspenSimulation.Tree.FindNode("\Data\Blocks\B3\Output\WNET").Value
    return w1+w2

sol = minimize(net_work, 4, method = 'Nelder-Mead')
print(sol)
print(sol.x)
```

     final_simplex: (array([[2.9       ],
           [2.89990234]]), array([5.73175519, 5.7317834 ]))
               fun: 5.7317551899999994
           message: 'Optimization terminated successfully.'
              nfev: 40
               nit: 17
            status: 0
           success: True
                 x: array([2.9])
    [2.9]
    
결과는 2.9가 나온다. 위에서 뽑았던 그림에서도 얼추 확인할 수 있듯이 적당한 최적값으로 보인다.



```python
AspenSimulation.Close()
```

위 코드는 프로그램을 정상적으로 종료해준다. 저장하고 종료하는 것도 가능하다. 궁금하면 찾아보자.

### 2.4 이미 있는 기능이었음

이 포스팅 준비하면서 알게 된 점이지만, Aspen Plus안에 이미 최적화 관련된 툴을 제공해준다. Model Analysis Tools 안에 Optimization기능이 이미 있다. 유튜브 영상은 [여길](https://www.youtube.com/watch?v=1KMfHHz2H4A) 보자. 솔직히 이런 기능이 있는지는 몰랐지만 있을 거 같긴 했다.

두 가지를 생각해 볼 수 있었다.

첫 번째는 일단 내가 제어하고 싶어하는 프로그램이 있다면 그게 어떤 기능이 있고 어떤 프로그램인지 알아두는 것이 첫 번째 인 것 같다. 굳이 정상적으로 잘 제공해주는 기능을 구현하느라 `win32com`이라는 모듈을 공부하면서 돌아갈 필요는 없을 것 같다. 그리고 프로그램 자체에서 제공해주는 기능을 사용하는 것이 대부분의 경우에 오류도 적고 시간도 더 빠를 것이다.

두 번째는 그렇다고 해서 위 코드들이나 알아본 것들이 의미가 없지는 않다는 점이다. 최적화 외에 무궁무진한 단순 반복적인 계산들을 잘만 쓰면 상당부분 해결해 줄 수 있다. 실제로 회사에서도 다른 일들로 사용했었는데 시간절약을 꽤나 할 수 있었다. 회사일들을 올리기는 어려워서 굳이 다른 예제를 찾아 정리해본 포스팅이었다.

## Summary

뭔가 좀 더 자동화스러운 자동화 해봤다. 어려웠고 알아봐야 할 것도 너무 많았지만 코드자체는 그리 길지 않았다. 만약에 다른 프로그램을 제어해보라고 한다면 쉽지 않을 것 같다.

먼저, 제어해야 할 프로그램 이름부터 (컴퓨터가 알아먹을 수 있는) 알아야 한다. ProgID.

그리고, 제어해야 할 프로그램 안의 특정 기능이나 변수에 접근할 수 있는 ID같은 것을 알아야 했다. Aspen Plus에서는 variable explorer로 알았지만, 다른 프로그램에서는 구글링아니면 혼자 알아보기는 힘들 수 있다. 그리고 ComObject에서 제공해주는 함수들 좀 알고나서 적절히 조합해 주면 됬다. 

자동화라는 걸 몇 번 해보다보면 생각보다 선택의 영역이라는 생각이 든다. 모든일을 오류없이 자동화 할 수는 없다. 그리고 자동화라는걸 적용하기에는 내가 투입해야 할 에너지와 시간이 적지 않다. 자동화 코드를 짜고 적용을 하더라도 결국 최종 확인과 코드의 유지보수도 나의 몫이다. 컴퓨터 화면 휙휙 움직이는거 보면 기분은 좀 좋을지 몰라도 모든 일을 이렇게 할 수는 없고 코드 외적으로 현실적으로 고려해야 할 부분이 있다. 잘만 쓰면 파워풀하지만 만능은 아니다.