---
title: "[Eng. Calc.] Symbolic Computation"
date: 2022-10-17 19:55:00 +0800
categories: [Engineering Calculation, Symbolic Computation]
tags: [Engineering Calculation, Python, Symbolic Computation]
math: true
mermaid: true
---

이번 공학용계산 포스팅의 주제는 Symbolic Computation, 기호 계산(?)이다. 개인적으로는 보통 그냥 심볼릭한 계산이라고 부른다. Symbolic Calculation이라고 검색해보려 했는데 구글에서 자동완성으로 Computation을 추천한다. 고민없이 Computation을 쓰자.

Symbolic한 계산이란 것은 인간이 수행하는 문자간의 연산, 미분, 적분, 라플라스 변환, 일반항에 대한 Summation 등을 생각하면 된다. $x$를 미분하면 1, 적분하면 ${1 \over 2}x^2 +C$ 또는 일반항의 합인 $\sum_{k=1}^n k = {1 \over 2}n(n+1)$ 이런 계산들을 컴퓨터한테 시키는 식이다. 계산 결과 또한 문자에 대한 식으로 나오게 된다. 더 자세한건 [여기](https://en.wikipedia.org/wiki/Computer_algebra)를 한번 읽어보자.

물론 컴퓨터는 이런 연산을 인간처럼 이해하고 하지는 않을 것이다. 사람이 일정한 형태로 짜놓은 명령어대로 움직일 것이다. 다만 이러한 일련의 연산들을 통해 사람이 하는 실수를 줄이거나 귀찮은 작업들을 컴퓨터가 해주는 장점들이 있을 수 있다.

일하면서는 쓸일이 없었지만 학생 시절에 많이 썼던 기능들이라 따로 소개한다. 알아둬서 나쁠건 없다. 이런게 있구나 하는 정도로 가볍게 넘겨도 좋다.

## 1. Intro

먼저 기억해야 할 점 하나는 컴퓨터를 활용한 Symbolic계산은 Numerical한 접근에 비해서 느리다는 점이다. 그럼에도 Symbolic한 계산을 사용하는 이유는 아주복잡한 식들을 '입력'하는데 사람이 일일이 유도하고 타이핑하는 작업을 피하기 위해서다. 물론 다른 이유도 많을 수 있지만 나는 그랬다.

예를 들어 아주 복잡한 식이 있고(초월함수, 다항함수의 조합에 분모에도 변수가 있지만 미분가능한) 이 식의 다양한 편미분식들을 사용해서 방정식 풀이를 한다고 가정해보자. 미분식들 하나하나 유도하고 입력하다가 지치게 된다. 운이좋아 한방에 잘 입력했으면 모르겠는데 (내 경험상) 항상 실수가 존재하고 이런경우 어디가 문제인지 찾기도 어렵다. 

이럴 때 근원이 되는 복잡한 식 하나만 신경써서 잘 입력해주고 나머지는 Symbolic한 계산을 통해 미분을 취해준다. 이후에 Numerical한 계산을 할 수 있게 변환해주고 필요한 계산을 하면 된다.

여담으로 적분 연산 결과는 해석해가 존재하지 않는 경우도 있고 적분상수 처리나 정적분의 경우 발산등 미분에 비해 일반적으로 신경쓰기 어려운 부분들이 있어 Symbolic한 계산으로 적분은 추천하지 않는다. 가능하면 제한을 많이두고 Numerical한 적분을 하길 바란다.

일단은 아래와 같이 간단한 예제를 한번 보자.
```python
from sympy import symbols, lambdify, diff
import numpy as np

x = symbols("x")
s = x**2 + x + 1
dsdx = diff(s, x)
dsdx_nu = lambdify(x, dsdx)

print(f"s = {s.doit()}")
print(f"dsdx = {dsdx.doit()}")
print(f"dsdx(x=1) = {dsdx_nu(1)}")
```

    s = x**2 + x + 1
    dsdx = 2*x + 1
    dsdx(x=1) = 3


`sympy`라는 Symbolic계산 전용 모듈을 사용한다. `s`라는 변수에 $x$에 대한 2차식을 정의했다. 그리고 `diff`명령어로 미분을 취하고 `dsdx`라는 변수에 지정해준다 `lambdify`명령어로 numerical한 계산을 위해 $x$에 대한 함수로 만들어 `dsdx_nu`라는 이름을 주었다. 계산결과는 생각한대로 잘 나온것을 볼 수 있다.

아래 조금 더 복잡한 예제도 한번 보자. 관련 코드는 [여기](https://stackoverflow.com/questions/26402387/sympy-summation-with-indexed-variable)서 따왔다.

```python
from sympy import Sum, symbols, Indexed, lambdify, diff
import numpy as np

a, x, i = symbols("a x i")
s = Sum(Indexed('x',i),(i,0,3))
print(f"s = {s.doit()}")

s2 = Sum(Indexed('x',i),(i,1,3))
print(f"s2 = {s2.doit()}")

s3 = Sum((a**i)*Indexed('x', i), (i, 0, 3))
print(f"s3 = {s3.doit()}")

f = lambdify(x, s)
f2 = lambdify(x, s2)
f3 = lambdify(x, s3)

print(type(f))
print(type(f2))
print(type(f3))

b = np.array([1, 2, 3, 4])

df3 = diff(f3(b), a)

print(f"f(b) = {f(b)}")
print(f"f2(b) = {f2(b)}")
print(f"f3(b) = {f3(b)}")
print(f"df3(b)/da = {df3}")

print(type(f(b)))
print(type(f2(b)))
print(type(f3(b)))
print(type(df3))

ff3 = lambdify(a, df3)

print(ff3(1))
```
    s = x[0] + x[1] + x[2] + x[3]
    s2 = x[1] + x[2] + x[3]
    s3 = a**3*x[3] + a**2*x[2] + a*x[1] + x[0]
    <class 'function'>
    <class 'function'>
    <class 'function'>
    f(b) = 10
    f2(b) = 9
    f3(b) = 4*a**3 + 3*a**2 + 2*a + 1
    df3(b)/da = 12*a**2 + 6*a + 2
    <class 'numpy.int32'>
    <class 'numpy.int32'>
    <class 'sympy.core.add.Add'>
    <class 'sympy.core.add.Add'>
    20

`indexed`라는 기능을 사용했는데 일반항처럼 idnex가 있는 계수들을 처리할 때 편해보인다. 먼저 사용할 symbol들을 정의하고 식을 정의했다. 아래 코드는 다음 식을 의미한다.

$$s = \sum_{i=0}^3 x_i $$

```python
s = Sum(Indexed('x',i),(i,0,3))
```
이를 `lambdify`로 x에 대한 함수를 만들어 `f`라고 했다. type을 확인해보면 function이라고 한다. numpy list b를 x에 대응해주면 함수 f는 1 + 2 + 3 + 4의 값을 return 한다. 미안하지만 다른 부분은 알아서 눈치껏 이해해보자. 보다 자세한 설명은 [Sympy Documents](https://docs.sympy.org/latest/index.html)를 확인해본다. 너무 무책임해 보이지만,, 어쩔 수 없다. 분량 조절해야된다.

일단 식을 정의하고 필요한 경우 미분이나 추가적인 심볼릭한 연산을 해준 다음에 수치해석적인 처리가 필요하게 되면 `lambdify`를 통해 type을 변환해주어 방정식 풀이나 숫자 대입을 하는 방식이다. 이 흐름만 기억해두자.

딱 여기까지만 봐도 좋다. 이정도만 봐도 필요할 때 언제든 갖다 쓸 수 있다.

이 밑의 예제는 훨씬 더 복잡한 상황에서 `sympy`를 적용해 보았다. 일단 코드를 짜고나니 아까워서 정리해보긴 했는데 아무리 생각해도 조금 과한것 같다.


## 2. Symbolic Computation for Phase Equilibrium

### 2.1. Helmholtz Free Energy Model

Symbolic Computation 예제로 이번 포스팅에서 고른 메인 주제는 Helmholtz Free Energy Model을 이용한 상평형 계산이다. 겪어본 가장 복잡한 식을 활용한 symbolic계산과 방정식 풀이의 조합 예제이다.

Cyclohexane이라는 물질에 대한 열역학 모델 식이며 원 논문은 [여기](https://www.nist.gov/system/files/documents/2018/03/13/10an_equation_of_state_for_the_thermodynamic_properties_of_cyclohexane.pdf)에서 받을 수 있다.

식이 매우 복잡한 만큼 현존하는 모델 중에 정확도가 가장 높고 예측가능한 온도, 압력 범위가 가장 넓다. Aspen+ 같은 공정모사 프로그램에서 REFPROP라는 열역학 모델로 사용할 수 있다. 단, 적용할 수 있는 물질들의 종류가 조금 제한적이다. 모델이나 계산 패키지에 대해 궁금하다면 [여기](https://www.nist.gov/srd/refprop)를 읽어보자.

Helmholtz Free Energy식은 아래와 같이 주어진다.

$${a(\rho, T) \over RT} = \alpha(\delta, \tau) = \alpha^0(\delta, \tau) + \alpha^r(\delta, \tau) \qquad (1)$$

$\delta$는 Reduced density로 $\rho / \rho_c$를 의미하며 $\tau$는 **inverse** reduced temperature이며 $T_c/T$로 정의된다. ($\rho$는 밀도, $T$는 온도이고 밑첨자 c는 critical state를 의미한다.) R은 기체상수, $a$는 Helmholtz energy, $\alpha$는 reduced Helmholtz energy, 위첨자 0와 r은 각각 ideal과 residual property를 의미한다.

식(1)의 ideal term, $\alpha^0$는 아래와 같이 주어져 있다.

$$\alpha^0 = a_1 + a_2\tau + \ln\delta + (c_0 - 1)\ln\tau + \sum_{k=1}^4 v_k\ln\left(1-\exp(-u_k \tau / T_c)\right) \qquad (2)$$

파라미터의 값은 $c_0$ = 4, $u_1$ = 773, $u_2$ = 941, $u_3$ = 2185, $u_4$ = 4495, $v_1$ = 0.83775, $v_2$ = 16.036, $v_3$ = 24.636, $v_4$ = 7.1715, $a_1$ = 0.9891140602, $a_2$ = 1.6359660572 이라고 한다.

남아있는 residual term, $\alpha^r$은 아래와 같이 정의된다.

$$\begin{align*}

\alpha^r(\delta, \tau) &= \sum_{i=1}^5 n_i \delta^{d_i} \tau^{t_i} + \sum_{i=6}^{10} n_i \delta^{d_i} \tau^{t_i} \exp (-\delta^{l_i})\\
& +\sum_{i=11}^{20}n_i \delta^{d_i} \tau^{t_i} \exp(-\eta_i(\delta - \varepsilon_i)^2 - \beta_i(\tau - \gamma_i)^2) \qquad (3)

\end{align*}$$

이 경우 파라미터는 아래 테이블로 주어진다.


|$i$ | $n_i$ | $t_i$ |$d_i$ |$l_i$ |$\eta_i$ |$\beta_i$ |$\gamma_i$ |$\varepsilon_i$ |
|:--:|:--|:--|:--|:--|:--|:--|:--|:--|
|1|0.05483581|1.00|4|0|0|0|0|0|
|2|1.607734|0.37|1|0|0|0|0|0|
|3|-2.375928|0.79|1|0|0|0|0|0|
|4|-0.5137709|1.075|2|0|0|0|0|0|
|5|0.1858417|0.37|3|0|0|0|0|0|
|6|-0.9007515|2.4|1|2|0|0|0|0|
|7|-0.5628776|2.5|3|2|0|0|0|0|
|8|0.2903717|0.5|2|1|0|0|0|0|
|9|-0.3279141|3|2|2|0|0|0|0|
|10|-0.03177644|1.06|7|1|0|0|0|0|
|11|0.8668676|1.6|1|0|0.99|0.38|0.65|0.73|
|12|-0.1962725|0.37|1|0|1.43|4.2|0.63|0.75|
|13|-0.1425992|1.33|3|0|0.97|1.2|1.14|0.48|
|14|0.004197016|2.5|3|0|1.93|0.9|0.09|2.32|
|15|0.1776584|0.9|2|0|0.92|1.2|0.56|0.2|
|16|-0.04433903|0.5|2|0|1.27|2.6|0.4|1.33|
|17|-0.03861246|0.73|3|0|0.87|5.3|1.01|0.68|
|18|0.07399692|0.2|2|0|0.82|4.4|0.45|1.11|
|19|0.02036006|1.5|3|0|1.4|4.2|0.85|1.47|
|20|0.00272825|1.5|2|0|3|25|0.86|0.99|

대략 160개 정도의 파라미터가 전체 모델을 정의하는데 쓰인다. 

Helmholtz Free Energy에 대한 적당한 미분식들을 통해 여러가지 열역학 물성에 대한 식을 뽑아낼 수 있으며 상평형 계산에 필요한 식들도 여기서 나오게 된다. 이 포스팅에서 사용할 물성들은 아래 식으로 구할 수 있다.

$$Z = {p \over \rho R T} = 1 + \delta \left( {\partial \alpha^r \over \partial \delta}\right)_{\tau} \qquad (4)$$

$$p = R T \rho_c \delta \left[1 + \left( {\partial \alpha^r \over \partial \delta}\right)_{\tau} \right] \qquad (5)$$

$$ {G \over {RT}}= 1 + \alpha^0 + \alpha^r + \delta \left( {\partial \alpha^r \over \partial \delta}\right)_{\tau} \qquad (6) $$

물론 위 식들 이외에도 다양한 열역학 관계식들을 활용해서 Heat Capacity, Joule-Thompson Coefficient, Speed of Sound 등등 여러 물성들을 계산할 수도 있다. 일단 여기서는 위 식(4~6)만 사용해본다. 궁금하면 직접 해보자.

정리하자면 식(1)로 정리된 Helmholtz Energy를 식(4~6)에서 정의된 편미분 적용을 통해 최종 물성식을 뽑아내고 상평형 계산을 수행해 볼 것이다. 당연히 미분에는 `sympy`를 쓰고 방정식 풀이를 위해 `lambdify`와 공학용계산 모듈인 `scipy`를 쓸 것이다.

상평형 계산은 임의의 온도 $\tau$에서 아래 두개의 방정식(equal pressure, equal gibbs energy)을 만족시키는 두 개의 density $\delta'$, $\delta''$를 구하게 한다.

$$p(\tau, \delta') = p(\tau, \delta'') \qquad (7)$$

$$G(\tau, \delta') = G(\tau, \delta'') \qquad (8)$$



### 2.2. Python Code

파라미터가 160개 가량 되는 모델식을 코드로 입력해야 하는 것 뿐만 아니라 이 식을 손으로 미분까지해서 또 새로 코드를 작성한다는 것은 생산성도 떨어질 뿐더러 실수할 가능성도 너무 많다. (물론, 이런일을 해야 할 상황이 인생에서 많지는 않다.)

어찌됬든 `sympy`를 이용하는 방식이 이러한 상황에서 그나마 가장 효율적이고 실수도 덜한 방법의 조합이라고 생각한다. 

아래 코드도 몇 번이나 실수한 끝에 겨우 성공한 것이기도 하다. 이 모듈이 없었다면, 시도조차 안했을 것 같다.

```python
import sympy as sp
import numpy as np
from scipy.optimize import fsolve
```
사용할 모듈은 `sympy`가 메인이다. 다른모듈은 다른 포스팅에서 이미 다루었다.

```python
delta, tau, i, v, u, n, t, d, l, eta, beta, gamma, epsilon = sp.symbols("delta tau i v u n t d l eta beta gamma epsilon")

vi = sp.Indexed('v', i)
ui = sp.Indexed('u', i)
ni = sp.Indexed('n', i)
ti = sp.Indexed('t', i)
di = sp.Indexed('d', i)
li = sp.Indexed('l', i)
etai = sp.Indexed('eta', i)
betai = sp.Indexed('beta', i)
gammai = sp.Indexed('gamma', i)
epsiloni = sp.Indexed('epsilon', i)

Tc = 553.6      # kelvin
rhoc = 3.224    # mol/dm-3
R = 0.008314    # MPa dm-3 / mol K
Mw = 84.15948   # g mol-1

# <------------------------------------------- Define Ideal Helmholtz Energy ------------------------------------------------> #
a1 = 0.9891140602
a2 = 1.6359660572
c0 = 4

u_value = np.array([0, 773, 941, 2185, 4495])
v_value = np.array([0, 0.83775, 16.036, 24.636, 7.1715])

ideal = a1 + a2*tau + sp.log(delta) + (c0 - 1)*sp.log(tau) + sp.Sum(vi*sp.log(1-sp.exp(-ui*tau/Tc)), (i, 1, 4))

# <------------------------------------------- Define Residual Helmholtz Energy ------------------------------------------------> #

n_value = np.array([0, 0.05483581, 1.607734, -2.375928, -0.5137709, 0.1858417, -0.9007515, -0.5628776, 0.2903717, -0.3279141, -0.03177644, 0.8668676, -0.1962725, -0.1425992, 0.004197016, 0.1776584, -0.04433903, -0.03861246, 0.07399692, 0.02036006, 0.00272825])
t_value = np.array([0, 1, 0.37, 0.79, 1.075, 0.37, 2.4, 2.5, 0.5, 3, 1.06, 1.6, 0.37, 1.33, 2.5, 0.9, 0.5, 0.73, 0.2, 1.5, 1.5])
d_value = np.array([0, 4, 1, 1, 2, 3, 1, 3, 2, 2, 7, 1, 1, 3, 3, 2, 2, 3, 2, 3, 2])
l_value = np.array([0, 0, 0, 0, 0, 0, 2, 2, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
eta_value = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.99, 1.43, 0.97, 1.93, 0.92, 1.27, 0.87, 0.82, 1.4, 3])
beta_value = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.38, 4.2, 1.2, 0.9, 1.2, 2.6, 5.3, 4.4, 4.2, 25])
gamma_value = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.65, 0.63, 1.14, 0.09, 0.56, 0.4, 1.01, 0.45, 0.85, 0.86])
epsilon_value = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.73, 0.75, 0.48, 2.32, 0.2, 1.33, 0.68, 1.11, 1.47, 0.99])

residual1 = sp.Sum(ni*(delta**di)*(tau**ti),(i, 1, 5))
residual2 = sp.Sum(ni*(delta**di)*(tau**ti)*sp.exp(-delta**li), (i, 6, 10))
residual3 = sp.Sum(ni*(delta**di)*(tau**ti)*sp.exp(-etai*((delta-epsiloni)**2)-betai*((tau - gammai)**2)), (i, 11, 20))

residual = residual1 + residual2 + residual3

# <---------------------------- Differentiate Helmholtz Energy to obtain specific property ----------------------------------> #

Z = 1 + delta*sp.diff(residual, delta)
P = Z*R*(Tc/tau)*delta*rhoc
G = 1 + ideal + residual + delta*sp.diff(residual, delta)
```
사용해야 할 symbol들의 종류도 워낙 많아서 좀 헷갈릴 수 있다. Intro에서 다루었던 예제를 생각해 보면 그래도 금방 이해될 것이다. 최종적으로 식(1)에 해당하는 Helmholtz Free Energy를 `residual`이라는 변수로 정의했고 식(4~6)에 해당하는 미분을 `sympy.diff`명령어 한줄로 끝낸다. 

식(1)만 신경써서 작성하면 된다. 여지없이 여기에서 계속 실수를 했었다.

```python
def P_num(delta_n, tau_n):
    P_inter = sp.lambdify((v, u, n, t, d, l, eta, beta, gamma, epsilon, delta, tau), P)
    return P_inter(v_value, u_value, n_value, t_value, d_value, l_value, eta_value, beta_value, gamma_value, epsilon_value, delta_n, tau_n)

def G_num(delta_n, tau_n):
    G_inter = sp.lambdify((v, u, n, t, d, l, eta, beta, gamma, epsilon, delta, tau), G)
    return G_inter(v_value, u_value, n_value, t_value, d_value, l_value, eta_value, beta_value, gamma_value, epsilon_value, delta_n, tau_n)

def equil_eqn(x):
    delta_a = x[0]
    delta_b = x[1]
    equal_pres = P_num(delta_a, tau_set) - P_num(delta_b, tau_set)
    equal_G = G_num(delta_a, tau_set) - G_num(delta_b, tau_set)
    return [equal_pres, equal_G]
```
상평형 계산에 사용할 함수들을 정의한다. 수치해석 계산을 위해 `sympy`식들을 `lambdify`를 통해 일종의 형태(type)변환을 해준다. `equil_eqn`함수가 `fsolve`를 적용해 풀어줘야 할 방정식에 해당한다.

```python
den_a = 9.37
den_b = 0.0027

delta_a = den_a/rhoc
delta_b = den_b/rhoc

temp_set = 283
tau_set = Tc/temp_set

root_eq = fsolve(equil_eqn, [delta_a, delta_b])

root = []
root.append(root_eq)
temp_list = [temp_set]
```


```python
temp_start = temp_set
step = 10
for i in range(1, int((round(Tc)-temp_start)/step) + 2, 1):
    temp_set = temp_set + step
    if temp_set >= Tc:
        break
    temp_list.append(temp_set)
    tau_set = Tc/temp_set
    root_eq = fsolve(equil_eqn, root[i-1])
    root.append(root_eq)
```
이전 포스팅에서와 비슷한 느낌으로 이전계산에서 얻은 답을 다음번 계산의 초기값으로 넣어주는 방식이다. 

```python
den_a_list = np.array(root)[:,0]*rhoc
den_b_list = np.array(root)[:,1]*rhoc
pres_list = P_num(np.array(root)[:,0],Tc/np.array(temp_list))

import matplotlib.pyplot as plt

temp_data = np.array([283,313,343,373,403,433,463,493,523,553])
den_a_data = np.array([9.3644,9.0275,8.6788,8.314,7.927,7.5071,7.0341,6.4677,5.7026,3.721])
den_b_data = np.array([0.002687,0.0095389,0.026088,0.059385,0.11875,0.21701,0.37422,0.62851,1.08,2.7321])
pres_data = np.array([0.0062923,0.024494,0.072215,0.17403,0.3613,0.67039,1.1417,1.8204,2.7605,4.0495])

plt.title('Saturated Density')
plt.plot(den_a_list, temp_list)
plt.plot(den_b_list, temp_list)
plt.scatter(den_a_data, temp_data)
plt.scatter(den_b_data, temp_data)
plt.xlabel('Molar Density(mol/l)')
plt.ylabel('Temperature(K)')
plt.show()

plt.title('Vapor Pressure')
plt.plot(temp_list, pres_list)
plt.scatter(temp_data, pres_data)
plt.xlabel('Tmmperature(K)')
plt.ylabel('Pressure(MPa')
plt.show()
```
Data는 [NIST Webbook](https://webbook.nist.gov/chemistry/fluid/)에서 따왔고 temp_data, den_a_data, den_b_data, pres_data에 해당한다. 결과는 매우 정확하다. (애초에 NIST Webbook은 이 모델의 결과를 알려준다.)

    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-10-18/output_6_0.png)


![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-10-18/output_6_1.png)

 

## 3. Advanced Algorithm

위에서 다룬 계산은 상당히 복잡한 비선형 방정식 2개로 이루어진 시스템에 대한 풀이에 해당한다. 한방에 모든 방정식을 풀려고 할 때 수치해석적인 풀이를 위해서는 답에 거의 근접한 초기값을 넣어주어야 한다. 그래서 보통 이러한 계산은 초기값에 대한 민감성을 덜어주기 위해 다양한 방식의 successive substitution 알고리즘이 적용된다.

Ryo Akasaka라는 분이 쓰신 이 [논문](https://www.jstage.jst.go.jp/article/jtst/3/3/3_3_442/_article)에 나온 알고리즘을 적용해 본다. 나름 이해하기 쉽게 작성이 되어 있어서 관심있으면 읽어보는 것도 좋을 것 같다.

실은 위에 애써 짜놓은 코드들 그냥 버리기 아까워서 이거조금 더 써먹어 본다.


### 3.1. Newton-Raphson algorithm for simultaneous equations

풀어야 하는 방정식이 여러개 있는 경우에 적용되는 Newton-Raphson 형태의 수치해석 알고리즘을 적용한다. 잠시 논문내용을 정리해본다.

Ideal Helmholtz term에 해당하는 식(2)는 아래처럼 표현할 수 있다.

$$\alpha^0 = \ln\delta + f(\tau) \qquad (9)$$

여기서 $f$ 는 온도만의 함수에 해당한다.

따라서 상평형 조건에 해당하는 식(7~8)을 정리하면 아래와 같이 주어진다.

$$\delta' \left[1 + \left( {\partial \alpha^r \over \partial \delta}\right)_{\tau, \delta = \delta'} \right] = \delta'' \left[1 + \left( {\partial \alpha^r \over \partial \delta}\right)_{\tau, \delta = \delta''} \right] \qquad (10)$$

$$\ln\delta' + \alpha^r(\tau, \delta') + \delta' \left( {\partial \alpha^r \over \partial \delta}\right)_{\tau, \delta = \delta'}  = \ln\delta'' + \alpha^r(\tau, \delta'') + \delta'' \left( {\partial \alpha^r \over \partial \delta}\right)_{\tau, \delta = \delta''} \qquad (11)$$

$J(\tau, \delta)$와 $K(\tau, \delta)$를 아래와 같이 정의하자.

$$J(\tau, \delta) = \delta \left[{1 + \left( {\partial \alpha^r \over \partial \delta}\right)_{\tau}}\right] \qquad (12)$$

$$K(\tau, \delta) = \ln\delta + \alpha^r(\tau, \delta) + \delta \left( {\partial \alpha^r \over \partial \delta}\right)_{\tau} \qquad (13)$$

이러면 식 (10~11)은 최종적으로 아래와 같이 정리된다.

$$J(\tau, \delta') = J(\tau, \delta'') \qquad (14)$$

$$K(\tau, \delta') = K(\tau, \delta'') \qquad (15)$$

임의의 $\tau$ 에서 두 개의 미지수 $\delta'$와 $\delta''$는 위 식 두개를 풀면서 얻어지게 된다.

이 부분에서 Newton-Raphson algorithm for simultaneous equations 를 적용하면 아래와 같이 successive substitution 형태의 식을 얻게된다.

$$\delta'^{(k+1)} = \delta'^{(k)} + {\gamma \over \Delta}\left\{ (K(\tau, \delta'') - K(\tau, \delta'))\left({\partial J \over 
\partial \delta}\right)_{\tau, \delta = \delta''} - (J(\tau, \delta'') - J(\tau, \delta'))\left({\partial K \over 
\partial \delta}\right)_{\tau, \delta = \delta''} \right\}  \qquad (16)$$


$$\delta''^{(k+1)} = \delta''^{(k)} + {\gamma \over \Delta}\left\{ (K(\tau, \delta'') - K(\tau, \delta'))\left({\partial J \over 
\partial \delta}\right)_{\tau, \delta = \delta'} - (J(\tau, \delta'') - J(\tau, \delta'))\left({\partial K \over 
\partial \delta}\right)_{\tau, \delta = \delta'} \right\}  \qquad (17)$$

여기서 $\Delta$는 아래와 같다.

$$\Delta = \left({\partial J \over \partial \delta}\right)_{\tau, \delta = \delta''} \left({\partial K \over \partial \delta}\right)_{\tau, \delta = \delta'} - \left({\partial J \over \partial \delta}\right)_{\tau, \delta = \delta'} \left({\partial K \over \partial \delta}\right)_{\tau, \delta = \delta''} \qquad (18)$$

Step size를 의미하는 $\gamma$는 특수한 경우가 아닌이상 1로 둔다.

최종적으로 아래의 Convergence Criteria를 사용했다.

$$\left( K(\tau, \delta'') - K(\tau, \delta') \right)^2 + \left( J(\tau, \delta'') - J(\tau, \delta') \right)^2 < 10^{-20} \qquad (19)$$

복잡해 보이지만 그냥 미분좀 더 하면 된다.


### 3.2. Python Code

2장에서 사용한 Code와 거의 비슷한 흐름이다. 

```python
import sympy as sp
import numpy as np
from scipy.optimize import fsolve
```

`fsolve`함수는 사용되지 않는다. 실수로 안지웠다.


```python
delta, tau, i, v, u, n, t, d, l, eta, beta, gamma, epsilon = sp.symbols("delta tau i v u n t d l eta beta gamma epsilon")

vi = sp.Indexed('v', i)
ui = sp.Indexed('u', i)
ni = sp.Indexed('n', i)
ti = sp.Indexed('t', i)
di = sp.Indexed('d', i)
li = sp.Indexed('l', i)
etai = sp.Indexed('eta', i)
betai = sp.Indexed('beta', i)
gammai = sp.Indexed('gamma', i)
epsiloni = sp.Indexed('epsilon', i)

Tc = 553.6      # kelvin
rhoc = 3.224    # mol/dm-3
R = 0.0083144621    # MPa dm-3 / mol K
Mw = 84.15948   # g mol-1

# <------------------------------------------- Define Ideal Helmholtz Energy ------------------------------------------------> #
a1 = 0.9891140602
a2 = 1.6359660572
c0 = 4

u_value = np.array([0, 773, 941, 2185, 4495])
v_value = np.array([0, 0.83775, 16.036, 24.636, 7.1715])

ideal = a1 + a2*tau + sp.log(delta) + (c0 - 1)*sp.log(tau) + sp.Sum(vi*sp.log(1-sp.exp(-ui*tau/Tc)), (i, 1, 4))

# <------------------------------------------- Define Residual Helmholtz Energy ------------------------------------------------> #

n_value = np.array([0, 0.05483581, 1.607734, -2.375928, -0.5137709, 0.1858417, -0.9007515, -0.5628776, 0.2903717, -0.3279141, -0.03177644, 0.8668676, -0.1962725, -0.1425992, 0.004197016, 0.1776584, -0.04433903, -0.03861246, 0.07399692, 0.02036006, 0.00272825])
t_value = np.array([0, 1, 0.37, 0.79, 1.075, 0.37, 2.4, 2.5, 0.5, 3, 1.06, 1.6, 0.37, 1.33, 2.5, 0.9, 0.5, 0.73, 0.2, 1.5, 1.5])
d_value = np.array([0, 4, 1, 1, 2, 3, 1, 3, 2, 2, 7, 1, 1, 3, 3, 2, 2, 3, 2, 3, 2])
l_value = np.array([0, 0, 0, 0, 0, 0, 2, 2, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
eta_value = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.99, 1.43, 0.97, 1.93, 0.92, 1.27, 0.87, 0.82, 1.4, 3])
beta_value = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.38, 4.2, 1.2, 0.9, 1.2, 2.6, 5.3, 4.4, 4.2, 25])
gamma_value = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.65, 0.63, 1.14, 0.09, 0.56, 0.4, 1.01, 0.45, 0.85, 0.86])
epsilon_value = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.73, 0.75, 0.48, 2.32, 0.2, 1.33, 0.68, 1.11, 1.47, 0.99])

residual1 = sp.Sum(ni*(delta**di)*(tau**ti),(i, 1, 5))
residual2 = sp.Sum(ni*(delta**di)*(tau**ti)*sp.exp(-delta**li), (i, 6, 10))
residual3 = sp.Sum(ni*(delta**di)*(tau**ti)*sp.exp(-etai*((delta-epsiloni)**2)-betai*((tau - gammai)**2)), (i, 11, 20))

residual = residual1 + residual2 + residual3

# <---------------------------- Differentiate Helmholtz Energy to obtain specific property ----------------------------------> #

Z = 1 + delta*sp.diff(residual, delta)
P = Z*R*(Tc/tau)*delta*rhoc
G = 1 + ideal + residual + delta*sp.diff(residual, delta)

J_s = delta*(1 + delta*sp.diff(residual, delta))
K_s = sp.log(delta) + residual + delta*sp.diff(residual, delta)

delJ_s = sp.diff(J_s, delta)
delK_s = sp.diff(K_s, delta)
```
2장에서 추가로 $J$, $K$와 이들의 편미분에 대한 식만 더 추가했다. 미분처리는 정말 편하다.


```python
def J_n(delta_n, tau_n):
    J_inter = sp.lambdify((n, t, d, l, eta, beta, gamma, epsilon, delta, tau), J_s)
    return J_inter(n_value, t_value, d_value, l_value, eta_value, beta_value, gamma_value, epsilon_value, delta_n, tau_n)

def K_n(delta_n, tau_n):
    K_inter = sp.lambdify((n, t, d, l, eta, beta, gamma, epsilon, delta, tau), K_s)
    return K_inter(n_value, t_value, d_value, l_value, eta_value, beta_value, gamma_value, epsilon_value, delta_n, tau_n)

def delJ_n(delta_n, tau_n):
    delJ_inter = sp.lambdify((n, t, d, l, eta, beta, gamma, epsilon, delta, tau), delJ_s)
    return delJ_inter(n_value, t_value, d_value, l_value, eta_value, beta_value, gamma_value, epsilon_value, delta_n, tau_n)

def delK_n(delta_n, tau_n):
    delK_inter = sp.lambdify((n, t, d, l, eta, beta, gamma, epsilon, delta, tau), delK_s)
    return delK_inter(n_value, t_value, d_value, l_value, eta_value, beta_value, gamma_value, epsilon_value, delta_n, tau_n)
```
마찬가지로 `lambdify`이용한 type 변환을 해준다. 이제 숫자를 더 빠르게 뽑아낼 수 있다.

```python
def advanced_eqili(tau_test, delta_a, delta_b):
    for i in range(50):
        ddelta = delJ_n(delta_b, tau_test)*delK_n(delta_a, tau_test) - delJ_n(delta_a, tau_test)*delK_n(delta_b, tau_test)
        delta_a = delta_a + (1/(ddelta))*((K_n(delta_b, tau_test) - K_n(delta_a, tau_test))*delJ_n(delta_b, tau_test) - (J_n(delta_b, tau_test) - J_n(delta_a, tau_test))*delK_n(delta_b, tau_test))
        delta_b = delta_b + (1/(ddelta))*((K_n(delta_b, tau_test) - K_n(delta_a, tau_test))*delJ_n(delta_a, tau_test) - (J_n(delta_b, tau_test) - J_n(delta_a, tau_test))*delK_n(delta_a, tau_test))
        criteria = ((K_n(delta_b, tau_test) - K_n(delta_a, tau_test))**2 + (J_n(delta_b, tau_test) - J_n(delta_a, tau_test))**2)
        if criteria <= 10**-20:
            break
    return [delta_a, delta_b]
```
Newton-Raphson방식으로 얻어지는 successive substitution형태를 `advanced_equil`로 정의한다. 식(16~18)에 해당한다.

이후로는 완전히 동일하다. `fsolve`가 `advanced_equil`함수로 치환된 형태다.

```python
den_a = 9.3991
den_b = 2.31*10**-3

delta_a = den_a/rhoc
delta_b = den_b/rhoc

temp_set = 283
tau_set = Tc/temp_set

root_eq = advanced_eqili(tau_set, delta_a, delta_b)

root = []
root.append(root_eq)
temp_list = [temp_set]
```


```python
temp_start = temp_set
step = 10
for i in range(1, int((round(Tc)-temp_start)/step) + 2, 1):
    temp_set = temp_set + step
    if temp_set >= Tc:
        break
    temp_list.append(temp_set)
    tau_set = Tc/temp_set
    root_eq = advanced_eqili(tau_set, root[i-1][0], root[i-1][1])
    root.append(root_eq)
```

```python
den_a_list = np.array(root)[:,0]*rhoc
den_b_list = np.array(root)[:,1]*rhoc
pres_list = R*np.array(temp_list)*rhoc*J_n(np.array(root)[:,0], Tc/np.array(temp_list))

import matplotlib.pyplot as plt

temp_data = np.array([283,313,343,373,403,433,463,493,523,553])
den_a_data = np.array([9.3644,9.0275,8.6788,8.314,7.927,7.5071,7.0341,6.4677,5.7026,3.721])
den_b_data = np.array([0.002687,0.0095389,0.026088,0.059385,0.11875,0.21701,0.37422,0.62851,1.08,2.7321])
pres_data = np.array([0.0062923,0.024494,0.072215,0.17403,0.3613,0.67039,1.1417,1.8204,2.7605,4.0495])

plt.title('Saturated Density')
plt.plot(den_a_list, temp_list)
plt.plot(den_b_list, temp_list)
plt.scatter(den_a_data, temp_data)
plt.scatter(den_b_data, temp_data)
plt.xlabel('Molar Density(mol/l)')
plt.ylabel('Temperature(K)')
plt.show()

plt.title('Vapor Pressure')
plt.plot(temp_list, pres_list)
plt.scatter(temp_data, pres_data)
plt.xlabel('Tmmperature(K)')
plt.ylabel('Pressure(MPa')
plt.show()
```
결과 그림은 굳이 필요 없을 것 같다. 2장의 결과와 동일하다. 코드를 직접 가지고 조금 놀다보면 초기값에 좀 덜 민감한 느낌을 받기는 한다. 

다만 계산시간이 오래걸린다. `advanced_eqili`함수를 정의할 때 for문으로 작성한 부분이 아마도 문제일 것으로 보인다. Vectorized 형태로 쓰거나 좀 더 빠르게 돌아가는 형태의 코드를 작성해보면 좋겠지만, 아직은 잘 모르겠다. 어찌됬든 돌아는 가고 답은 나온다. 지금은 기능 구현에 집중하자.


## Summary

`sympy`를 활용한 공학용 계산을 다루었다. 포스팅 작성하면서 구글링 해볼때 심볼릭한 계산을 다루는 python 예제에서는 거의 예외없이 `sympy`가 사용되었던 것 같다.

일단 식을 정의하고 필요한 경우 미분이나 추가적인 심볼릭한 연산을 해준 다음에 수치해석적인 처리가 필요하게 되면 `lambdify`를 통해 type을 변환해주어 방정식 풀이나 숫자 대입을 하는 방식이었다. 이 흐름만 기억해두자. 다른 복잡한 식들은 그냥 다 잊어도 좋을 것 같다.

예제를 잘못 골랐는지 이번 글은 분량조절도 실패하고 개인적으로 너무 힘들었다.

