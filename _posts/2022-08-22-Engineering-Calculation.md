---
title: "[Eng. Calc.] Differential Equations - Kinetics"
date: 2022-08-22 19:55:00 +0800
categories: [Engineering Calculation, ODE-Kinetics]
tags: [Engineering Calculation, Python, Kinetics, Differential Equation]
math: true
mermaid: true
---

공학용 계산 첫 번째 포스팅이다. 첫번째 주제는 반응속도식 문제 풀이다. 수학으로는 미분방정식 풀이와 fitting 예제에 해당한다. 계산에는 Python을 쓴다. 

## 1. Intro

간단한 문제에서부터 시작하려고 한다. [이 문서](https://symfit.readthedocs.io/en/stable/examples/ex_ODEModel.html)의 예제와 데이터를 따라가본다. 아래 반응을 보자.

$$ A \rightarrow B $$

반응속도식을 정하는건 이런저런 고려들이 들어가는 복잡한 일이지만 일단 간단하게 반응차수는 1로 아래처럼 정리된다고 가정하자.

$$ {dC_A \over dt} = -k_1 C_A \qquad (1)$$

A라는 물질의 시간에 따른 농도변화는 A의 농도에 비례하며 비례상수는 반응속도상수인 $k_1$이다. A는 B로 변하면서 농도가 감소할 것이기 때문에 음수가 붙는다. 일반적으로 반응 실험 Data로부터 $k_1$을 Regression하고 다른 온도조건에서의 반응결과 예측이나 반응기 설계 등에 활용된다.

## 2. Background

식 1을 푸는 방식은 크게 2가지다. 손으로 미분방정식을 풀어서 해석해를 구하거나 수치적으로 근사해서 풀이한다. (Analytic solution을 구하거나 numerical solution을 구한다.)

### 2.1. Analytic Solution

식 (1)은 아래처럼 해석해가 구해진다.

$${dC_A \over C_A} = -k_1 dt,\quad\int_{C_{A,0}}^{C_A} {dC_A \over C_A} = \int_{0}^{t} -k_1 dt$$

$$C_A = C_{A,0}e^{-k_1 t} \qquad (2)$$

$C_{A,0}$는 t=0 일때의 초기농도를 의미한다. 

Data는 시간(tdata)와 각 시간에 측정한 A물질의 농도(concentration)이다.

```python
time_data = np.array([0, 0.9184, 9.0875, 11.2485, 17.5255, 23.9993, 27.7949, 31.9783, 35.2118, 42.973, 46.6555, 50.3922, 55.4747, 61.827, 65.6603, 70.0939])

CA_data = np.array([0.906, 0.8739, 0.5622, 0.5156, 0.3718, 0.2702, 0.2238, 0.1761, 0.1495, 0.1029, 0.086, 0.0697, 0.0546, 0.0393, 0.0324, 0.026])
```
Data를 보니 $C_{A,0}$는 0.906이다. input은 time_data이고 output은 conc_data이니 간단한 nonlinear regression 문제로 바뀌었다. 아래 코드로 $k_1$을 구할 수 있다. (구글에 `python nonlinear regression` 검색해보자.)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


time_data = np.array([0, 0.9184, 9.0875, 11.2485, 17.5255, 23.9993, 27.7949, 31.9783, 35.2118, 42.973, 46.6555, 50.3922, 55.4747, 61.827, 65.6603, 70.0939])
CA_data = np.array([0.906, 0.8739, 0.5622, 0.5156, 0.3718, 0.2702, 0.2238, 0.1761, 0.1495, 0.1029, 0.086, 0.0697, 0.0546, 0.0393, 0.0324, 0.026])
```


```python
def C_A(t, k1):
    C_A0 = CA_data[0]
    return C_A0*np.exp(-k1*t)

para, pcov = curve_fit(C_A, time_data, CA_data)

print(f"k1 = {para}")
```

    k1 = [0.05078669]
    


```python
plt.title("Regressed Results")
plt.plot(time_data, C_A(time_data, para))
plt.scatter(time_data, CA_data)
plt.xlabel("Time")
plt.ylabel("Concentraion of A")
plt.show()
```
    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-22/output_2_0.png)
    

scipy모듈의 curve_fit함수를 사용한 간단한 regression이다. k1은 0.05가 나왔다. 

참고로 curve_fit은 squared error sum을 최소화 해야 할 오차함수로 쓰는 것으로 보인다. 자세한건 [Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)을 읽어보자.

### 2.2. Numerical Solution

이번에는 수치적으로 식(1)을 풀어보자. 아래 코드로 구할 수 있다. 

```python
import numpy as np
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt


time_data = np.array([0, 0.9184, 9.0875, 11.2485, 17.5255, 23.9993, 27.7949, 31.9783, 35.2118, 42.973, 46.6555, 50.3922, 55.4747, 61.827, 65.6603, 70.0939])
CA_data = np.array([0.906, 0.8739, 0.5622, 0.5156, 0.3718, 0.2702, 0.2238, 0.1761, 0.1495, 0.1029, 0.086, 0.0697, 0.0546, 0.0393, 0.0324, 0.026])
```


```python
def f(y, t, k): 
    C_A = y
    k1 = k
    dCAdt = -k1*(C_A)
    return dCAdt


def Conc(x,param):
    f2 = lambda y,t: f(y, t, param)
    CA = integrate.odeint(f2,CA_data[0],x)
    return CA


def f_resid(p):
    res = 0
    for i in range(len(time_data)):
        res = res + (Conc(time_data, p)[i] - CA_data[i])**2
    return res

guess = 0.01

c = optimize.minimize(f_resid, guess)
print("parameter values are ", c.x)
```

    parameter values are  [0.05078668]
    


```python
plt.title("Regressed Results")
plt.plot(time_data, Conc(time_data, c.x))
plt.scatter(time_data, CA_data)
plt.xlabel("Time")
plt.ylabel("Concentraion of A")
plt.show()
```


    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-22/output_2_1.png)
    
조금 더 복잡해 보인다. 먼저 강조하고 싶은 건 내 머리속에서 처음부터 끝까지 나온 코드가 절대 아니다. [이 문서](https://stackoverflow.com/questions/11278836/fitting-data-to-system-of-odes-using-python-via-scipy-numpy)를 참고했다. 그림 끼워맞추기 처럼 생각해도 좋다. 위 코드를 재사용하려면 이 코드의 이 부분이 어떤 역할을 하는지 눈치껏 보고 내 상황에 맞게 고치면 된다.

간단히 설명하자면 `f`함수는 식(1) 미분방정식 자체를 정의하고 `integrate.odeint`로 수치적분한 함수를 만들어 주고(식(2)에 해당할 것이다.) 수치적분한 함수와 데이터간의 오차를 `f_resid`로 정의했다. 여기서 오차는 오차제곱의 합으로 선정했다. 이후 `optimize.minimize`로 오차 함수를 최소화 하는 $k_1$을 구하도록 했고 결과는 똑같이 0.05가 나왔다. 

두가지 방식 중 어떤 방식이 더 좋다고 할 수 있을까. 그보다 어느게 좋은 방식인지 말하는 기준은 뭘까.

### 2.3. Analytic vs Numerical

해석해를 통해 미분방정식을 푸는 일은 일단 꽤나 직관적이다. 인과관계를 파악하기도 용이할 것이고 어딘가에서 실수가 있었다면 발견하고 고치기도 쉬울 것이다. 참고로 해석해를 구하는 건 프로그래밍으로도 가능하다. (Symbolic한 연산을 수행해주는 [`sympy` 라이브러리](https://docs.sympy.org/latest/modules/solvers/ode.html)가 있다.) 하지만 해석해가 존재하지 않는 미분방정식 또한 존재한다. 따라서 해석해를 구하는 방식은 일반적으로 모든 경우에 적용가능한 방법은 아니다.

수치적인 방법은 복잡한 미분방정식 풀이에 적당할 수 있다. 그렇다고 Euler Method같은 미분방정식 풀이 알고리즘을 꼭 알아야 할 필요는 없다. 라이브러리와 Solver들이 잘 갖추어져 있어 그냥 갖다쓰면 된다. 하지만 미분방정식 중에도 풀기 어려운 형태의 식들이 있다. [Stiffness](https://en.wikipedia.org/wiki/Stiff_equation)같은 수치해석적인 이슈가 있는 경우도 있다. 물론 그런것들을 해결해주는 라이브러리 또한 Python 어딘가에 존재할 것이다.

둘 중 뭐가 더 좋다고 말할 수는 없다. 개인적으로는 쉬운 문제인 경우 직접 해석적으로 풀어보고 복잡한 문제라면 어쩔수 없이 수치적인 방법을 쓰는게 좋다고 생각한다. 근데 일단 미분방정식을 풀이할 일 자체가 인생에서 그리 많지는 않다..

## 3. Application

훨씬 복잡한 미분방정식 풀이를 해보려고 한다. 당연하게 Numerical method 쓴다. 6개의 물질(A~E)이 포함된 아래 반응에 대한 반응속도 문제를 예제로 삼아보자.

$$ A + B \rightarrow C + D$$

$$A + C \rightarrow E$$

반응속도식은 아래와 같이 주어진다고 가정하자.

$${dC_A \over dt} = -k_1 C_A^2C_B - k_2C_AC_C \qquad (3.1)$$

$${dC_B \over dt} = -k_1 C_A^2C_B \qquad (3.2)$$

$${dC_C \over dt} = k_1 C_A^2C_B - k_2C_AC_C \qquad (3.3)$$

$${dC_D \over dt} = k_1 C_A^2C_B \qquad (3.4)$$

$${dC_E \over dt} = k_2C_AC_C \qquad (3.5)$$

식 (3.1) ~ (3.4)에서 $C_A$의 반응차수가 1이 아니라 2라는 것에 주의하자. 이번에는 미분방정식 5개인 ODEs System을 풀면서 동시에 Data에 맞는 $k_1$과 $k_2$를 fitting 해야 한다.

Intro에서 보여줬던 numerical solution 코드를 약간 수정하여 적용하면 아래처럼 된다.

```python
import numpy as np
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt
```


```python
time_data = np.array([0, 20, 40, 60, 90, 120, 180])/60

CAdata = np.array([~~~])
CBdata = np.array([~~~])
CCdata = np.array([~~~])
CDdata = np.array([~~~])
CEdata = np.array([~~~])

Data = np.array([CAdata, CBdata, CCdata, CDdata, CEdata])
```


```python
def f(y, t, k): 
    C_A = y[0]
    C_B = y[1]
    C_C = y[2]
    C_D = y[3]
    C_E = y[4]

    k1 = k[0]
    k2 = k[1]

    dCAdt = -k1*(C_A**2)*C_B - k2*(C_A)*(C_C)
    dCBdt = -k1*(C_A**2)*C_B
    dCCdt = k1*(C_A**2)*C_B - k2*(C_A)*(C_C)
    dCDdt = k1*(C_A**2)*C_B
    dCEdt = k2*(C_A)*(C_C)

    return [dCAdt, dCBdt, dCCdt, dCDdt, dCEdt]


def Conc(x,param):

    f2 = lambda y,t: f(y, t, param)
    r = integrate.odeint(f2,y0,x)
    CA = r[:,0]
    CB = r[:,1]
    CC = r[:,2]
    CD = r[:,3]
    CE = r[:,4]

    return [CA, CB, CC, CD, CE]


def f_resid(p):
    
    res = 0
    for i in range(Data.shape[0]):
        for j in range(Data.shape[1]):
            if Data[i][j]==0:
                continue
            else:
                res = res + ((Data[i][j] - Conc(time_data, p)[i][j])**2)
    return res

guess = [0.001, 0.001] 

y0 = Data[:,0] 

c = optimize.minimize(f_resid, guess)

print("parameter values are ", c.x)
```

    parameter values are  [ 0.14028538 -0.02505139]
    


```python
legend = ['CA', 'CB', 'CC', 'CD', 'CE']
for i in range(len(Data)):
    plt.scatter(time_data, Data[i,:], label = legend[i])
    plt.plot(time_data, Conc(time_data, c.x)[i])
plt.legend()
plt.xlabel("Time")
plt.ylabel("Conc")
plt.show()

plt.scatter(time_data, Data[4,:])
plt.plot(time_data, Conc(time_data, c.x)[4])
plt.title("CE Plot")
plt.show()
print(f"MSE = {err}")
```


    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-22/output_3_0.png)
        
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-22/output_3_1.png)


앞에서와 마찬가지로 `f`는 미분식을,  `Conc`는 수치적분된 함수를, `f_resid`는 최소화 해야할 오차함수를 나타낸다. `f_resid`에서 return 해주는 `res`를 보면 단순한 오차의 제곱합임을 알 수 있다.

`optimize.minimize` 함수로 `f_resid`를 최소화하는 $k_1$, $k_2$를 구해보면 각각 0.14, -0.025가 나온다. 물리적으로 두 속도상수는 양수가 나와야 한다. E농도 그래프의 경우 음수영역에서 움직인다. 수학적으로는 오차함수를 최소화하는 두 계수를 구했지만 물리적으로 맞지않는다. 생각보다 흔하게 발생하는 일이다.

### 3.1. Constraint for Parameter

이런 경우에는 사용한 solver인 `optimize.minimize`에서 $k_1$, $k_2$의 범위를 정해주는 기능이 있는 경우 사용해주면 된다. 아니면 아래처럼 간단한 트릭을 사용해 줄 수도 있다.

```python
def f(y, t, k): 
    C_A = y[0]
    C_B = y[1]
    C_C = y[2]
    C_D = y[3]
    C_E = y[4]

    k1 = k[0]
    k2 = k[1]

    dCAdt = -k1*(C_A**2)*C_B - (k2**2)*(C_A)*(C_C)
    dCBdt = -k1*(C_A**2)*C_B
    dCCdt = k1*(C_A**2)*C_B - (k2**2)*(C_A)*(C_C)
    dCDdt = k1*(C_A**2)*C_B
    dCEdt = (k2**2)*(C_A)*(C_C)

    return [dCAdt, dCBdt, dCCdt, dCDdt, dCEdt]
```
애초에 식을 정의할 때 $k_2$에 제곱을 해준다. 이러면 $k_2$로 무슨 값이 나오든 식에 적용할 때에는 양수가 된다. 이런 방법을 잘 활용하면 exponential로 정의해서 양수가 되게 한다거나 $1/(1+x^2)$ 같이 0과 1사이가 되게 한다거나 하는 것도 생각 할 수 있다. 어차피 식이 복잡해져도 계산은 컴퓨터가 해줄 것이다.

Fitting 결과는 다음과 같다.

```
parameter values are  [ 1.4959010e-01 -8.4175129e-08]
```

![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-22/output_3_2.png)
    
    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-22/output_3_3.png)

$k_2$ 결과값은 -0.00000008로 제곱하면 거의 0이다. 자연스럽게 시간에 따른 E의 농도는 0이될 것이다. 애초에 음수가 아니면 0이 나올 상황이었던 것 같다.

### 3.2. Absolute Error vs Relative Error

 A~D물질의 경우 나타나지 않는 현상이 E에서만 유독 나타나는 이유는 뭘까. 실험데이터가 틀렸을 수 있고 반응속도식 (3.1) ~ (3.5)가 틀렸을 수 있고 또 하나는 데이터 scale의 문제일 수도 있다.

일종의 Cost 함수인 `f_resid`는 오차제곱합, squared error sum이고 아래 식으로 표현된다.

$$Cost = \sum_i^m(y_i-f(x_i))^2 \qquad (4)$$

$y$는 Data, $f(x_i)$는 계산값이다. Data를 잘 보면 A~D 물질들의 시간에 따른 농도 데이터는 대부분 1보다 크다. 하지만 E물질의 경우 농도 데이터는 커봐야 0.1이다. 식(4)로 동일하게 Cost를 적용할 경우 아무래도 값이 큰 A~D에 더 잘 맞도록 $k_1$, $k_2$를 움직일 것이다. 

다시 말하면 Data의 Scale이 다른 것이고 이게 심할때는 일종의 왜곡이 일어나게 된다. 이런 경우 아래와 같이 [상대오차](https://www.scienceall.com/%EC%83%81%EB%8C%80%EC%98%A4%EC%B0%A8relative-error-2/), relative error를 Cost로 사용해 볼 수 있다.

$$Cost = \sum_i^m \left({y_i - f(x_i) \over y_i}\right)^2 \qquad (5)$$

위 식에는 오차의 분모부분에 다시한번 data를 넣어주어서 작은 값을 가지는 data에는 잘 맞도록 충분한 가중치를 주고 반대로 큰 값을 가지는 data에는 penalty를 주는 느낌이다.

`f_resid`를 수정하고 돌려보면 결과는 다음과 같다.

```python
def f_resid(p):
    
    res = 0
    for i in range(Data.shape[0]):
        for j in range(Data.shape[1]):
            if Data[i][j]==0:
                continue
            else:
                res = res + ((Data[i][j] - Conc(time_data, p)[i][j])/Data[i][j])**2
    return res
```
    parameter values are  [0.10749438 0.11924306]

![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-22/output_3_5.png)
    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-08-22/output_3_6.png)

결과는 훨씬 좋다. $k_1$, $k_2$도 물리적으로 말이 되고 E물질 농도에 대해 경향성을 놓치지 않으면서 다른 data에도 적당한 fit을 보여준다. 최종 식(3)에 사용하게 될 속도 상수는 아래처럼 주어진다.

$$k_1 = 0.1075, \quad k_2 = 0.0142$$

Scipy.optimize를 반드시 써야 하는건 아니다. Multiple ODE regression은 다른 예제도 있고 symfit이라는 라이브러리의 경우 조금 더 직관적으로 이해가 되게끔 구성이 되있기도 하다. [이 예제](https://symfit.readthedocs.io/en/stable/examples/ex_ode_system.html)도 충분히 활용해 볼 만 하다. 어쩌면 끔찍할거같지만 엑셀로도 가능할지 모른다. 

편한만큼 자유도가 떨어지기도 한다. 반대로 자유도가 높으면 할 수 있는건 많아지지만 신경쓸것도 많아진다.

머리가 고생하면 몸이 편하지만, 몸이 고생하면 머리가 편하다.

## Summary

공학용 계산 첫번째 시간으로 미분방정식 풀이와 fitting까지 해봤다. 해석해를 구해서 단순 regression만 해보기도 했고 numerical하게 계산해서 오차함수를 정의한 다음에 이를 최소화 하는 함수를 사용하기도 했다. 

반응속도식을 세우고 상수를 구한다는게 개별 반응의 특성마다 상황이 굉장히 다르고 수식을 세우기 나름인 부분도 커서 일반적인 해결방법에 대한 얘기를 하지는 못한다. 이런 얘기를 하려면 반응공학 수업을 통째로 들어야 하고 작성자 본인도 잘 모른다. 

이 포스팅에서는 미분방정식이 세워졌을 때 어떤 식으로 문제를 해결할 수 있는지 Python으로 보여주는걸 목표로 잡았고 그 예제로 반응속도식을 선정한 것 뿐이다. 하지만 애써 수식들이 세워졌더라도 fitting 하는 데에 신경써야 할 부분들이 적지 않음을 알 수 있다.

Python만이 이런 문제를 해결할 수 있는 건 아니지만 이만큼 또 편하게 시도해보고 답을 찾을 수 있는 프로그래밍 언어도 없다고 생각한다. 다음번 포스팅에는 여러가지 Regression과 방정식 풀이로 예제를 가져올 생각이다.