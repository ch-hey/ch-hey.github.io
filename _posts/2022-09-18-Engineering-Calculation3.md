---
title: "[Eng. Calc.] Nonlinear Equations Solving"
date: 2022-09-18 19:55:00 +0800
categories: [Engineering Calculation, Equation Solving]
tags: [Engineering Calculation, Python, Nonlinear Equation Solving]
math: true
mermaid: true
---

이번 공학용계산 포스팅의 주제는 Equation Solving, 방정식 풀이다. 

선형방정식에 대한 풀이는 손으로도 쉽게 풀리기 때문에 여기서는 비선형방정식에 대한 풀이를 다뤄본다. 복잡한 수치해석적인 방법에 대한 소개는 없다. Python으로 어떻게 비선형방정식을 푸는지, 어떤 모듈이 필요한지 계산 예시를 통해 소개한다. 

화학공학에서 종종 만나게 되는 상태방정식을 통한 물성 계산을 예로 들어본다.

## 1. Intro

상태방정식을 통해 순수물질의 물성을 계산해 보고자 한다. [Peng-Robison 상태방정식](https://pubs.usgs.gov/of/2005/1451/equation.html)을 사용할 것이다. 계산에 필요한 식들은 아래와 같이 주어져 있다.

$$ P = {RT \over V-b} - {a \alpha \over V^2 + 2bV - b^2} \qquad(1a)$$

$$ P = {\rho RT \over 1-\rho b} - {a \alpha \rho^2 \over 1 + 2b\rho - (\rho b)^2} \qquad(1b)$$

$$ a = 0.45724{R^2{T_c^2} \over P_c}, \ b = 0.07780{R T_c \over P_c}$$

$$ \alpha = [1 + (0.37464 + 1.54226\omega - 0.26992\omega^2)(1 - T_r^{0.5})]^2 $$

$$ A = {a \alpha P \over R^2T^2}, \ B = {b P \over RT} $$

in Polynomial form:

$$ Z^3 - (1 - B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0  \qquad(2)$$

Fugacity:

$$ \ln\phi = Z - 1 - \ln(Z-B) - {A \over 2\sqrt{2} B}\ln{Z + 2.414B \over Z - 0.414B} \qquad(3)$$

$P$는 압력, $T$는 온도, $V$는 부피, $\rho$는 밀도, $T_c$는 임계온도, $P_c$는 임계압력, $w$는 acentric factor, $Z$는 압축인자, $\phi$는 fugacity 이다. 

기체상수 R은 아래와 같이 사용할 단위에 맞게 조정해준다.

$$ 8.314 {J \over mol \ K} \ {Pa \ m^3 \over J} \ {bar \over 10^5Pa} \ {10^6 cm^3 \over m^3} = 83.14 {bar \ cm^3 \over mol \ K}$$

뭔가 복잡해보인다. 하지만 Python이 다 풀어줄 것이다. 우리가 해야 할 것은 어떤 수식을 풀어달라고 코드짜는 것뿐이다. 이거 하려면 결국 해결할 문제를 수학적으로 표현하고 이해하는 것이 필요하다. 상태방정식을 공부할 필요는 없다. 그저 저런 식들을 푸는데 이런 Python 모듈을 쓰는구나, 내가 풀 문제는 이거니까 이렇게 고쳐봐야지 이정도만 생각하면 충분하다.

## 2. Equation solving - Mono-Variable

### 2.1. Liquid density at specified Temperature & Pressure

간단한 상황을 생각해본다. 먼저 온도가 충분히 높고 압력이 충분히 낮아서 기체상으로만 존재하는 상태에서 물질의 밀도를 계산해보려고 한다. [NIST Data](https://webbook.nist.gov/chemistry/fluid/)에 따르면 n-Hexane은 447K, 5.1bar에서 기체로만 존재하며 밀도는 0.00015228 mol/cm<sup>3</sup> 라고 한다.

Peng-Robinson 상태방정식은 아래와 같이 순수물질의 물성을 요구한다.

- Critical Temperature ($T_c$): 507.82 K
- Critical Pressure ($P_c$): 30.441 bar
- Acentric factor ($w$): 0.30

식 (1b)를 보면 온도, 압력, 밀도간의 관계식이 주어져있다. 여기에 온도, 압력을 넣어주면 밀도만의 비선형 식으로 정리되며 이를 풀면 원하는 기체 밀도를 얻을 수 있다. 1개의 식이 1개의 변수에 대해서만 표현되어 있어 문제를 풀 수 있다.

### 2.2. Python Code

언제나 그렇듯 `numpy`, `scipy.optimize`, `matplotlib`이 사용된다. `scipy`모듈은 공학용 계산에서 빼놓을 수 없을 것 같다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
```

```python
R = 83.14 # P in bar, V in cm3/mol, T in kelvin
## Parameter for n-Hexane, Data from NIST
Tc = 507.82
Pc = 30.441
w = 0.30

a = 0.45724*(R**2)*(Tc**2)/Pc
b = 0.07780*R*Tc/Pc

def alpha(temp):
    return (1 + (0.37464 + 1.54226 * w - 0.26992 * (w**2))*(1-(temp/Tc)**0.5))**2

def pres(temp, den):
    return R*temp*den/(1-den*b)-a*alpha(temp)*(den**2)/(1+2*b*den-(den*b)**2)

tempp = 447
presp = 5.1

def pres_fsol(den):
    return (pres(tempp, den)-presp)

root_v = fsolve(pres_fsol, 0.001)

print(root_v)
print(f"solution is {0.00015228:.6f} mol/cm3, Calculation result is {root_v[0]:.6f} mol/cm3")

```

    [0.0001526]
    solution is 0.000152 mol/cm3, Calculation result is 0.000153 mol/cm3


위에서 정의한 `pres`라는 함수가 식(1b)와 동일한 식이다. `pres_fsol`은 식(1b)로 표현되는 압력에서 5.1bar를 빼준 값을 return 해주는 함수로 0이 되어야 할 방정식에 해당한다. 바로 밑에 `scipy.optimize`에 있는 `fsolve`라는 명령어로 0이 되어야 할 함수인 `pres_fsol`을 풀어주며 밀도값인 0.0001526이 얻어졌다.

그렇게 복잡하진 않다. 0이 되어야 하는 함수를 정의해주고 `fsolve` 명령어로 적절한 초기값을 넣어서 실행시켜주면 답이 얻어진다. 0이 되어야 할 함수, 초기값 이 두개만 기억하자.

초기값은 적당한 값을 넣어주는데 값에 대한 힌트가 전혀 없다면 0이 되어야 할 함수를 변수에 대해 plot해보면서 확인하는 것을 추천한다. (위 경우에는 `pres_fsol`을 `den`에 대해 plot해볼수 있을 것이다. 아래코드처럼 하면 된다.)

```python
den_plot = np.linspace(0,0.006,100)
plt.plot(den_plot, pres(tempp, den_plot))
plt.show()
```


## 3. Equation solving - Multi-Variable

비선형식 1개를 푸는 것을 해봤다. 이번엔 여러개를 동시에 풀어보자. Code는 정확히 똑같은 흐름으로 진행된다.


### 3.1. Phase Equilibrium Calculation for Pure Substance

상평형에 대한 계산은 알아야 할 여러 배경지식들이 있지만 일단 여기서는 복잡한 비선형방정식 시스템을 풀이하는 것을 목적으로 하므로 다 생략하고 어떤 식을 풀어야 하는지 상황만 설정해본다.

특정 온도에서 Hexane이 기상과 액상의 형태로 공존하는 압력이 존재하며 각 상에서의 밀도를 계산하고자 한다.

변수의 종류는 온도 $T$, 이때의 압력 $P$, 액상의 압축인자 $Z_L$, 기상의 압축인자 $Z_V$ 4개이다. 

input으로 온도가 주어지므로 unknonwn의 개수는 3개이며 이를 구하기 위해 아래 3개의 식이 활용된다.

Polynomial Form:

$$ Z_L^3 - (1 - B)Z_L^2 + (A - 3B^2 - 2B)Z_L - (AB - B^2 - B^3) = 0  \qquad(4)$$

$$ Z_V^3 - (1 - B)Z_V^2 + (A - 3B^2 - 2B)Z_V - (AB - B^2 - B^3) = 0  \qquad(5)$$

Equal Fugacity:


$$ \ln\phi_L = \ln\phi_V \qquad (6)$$

$$ \ln\phi_L = Z_L - 1 - \ln(Z_L-B) - {A \over 2\sqrt{2} B}\ln{Z_L + 2.414B \over Z_L - 0.414B} $$

$$ \ln\phi_V = Z_V - 1 - \ln(Z_V-B) - {A \over 2\sqrt{2} B}\ln{Z_V + 2.414B \over Z_V - 0.414B} $$

이렇게 구해진 변수들로 액상, 기상의 밀도도 계산할 수 있다.

$$\rho_L = {P \over Z_LRT}, \ \rho_V = {P \over Z_VRT}$$

### 3.2. Python Code

위에서 사용한 Code에서 이어서 쓴다.



```python
def fugz(temp, z, p):
    A = a*alpha(temp)*p/((R**2)*(temp**2))
    B = b*p/(R*temp)
    return z-1-np.log(z-B)-(A/(2*(2**0.5)*B))*np.log((z + 2.414*B)/(z-0.414*B))

def zcubic(temp, z, p):
    A = a*alpha(temp)*p/((R**2)*(temp**2))
    B = b*p/(R*temp)
    return (z**3)-(1-B)*(z**2)+(A-3*(B**2)-2*B)*z-(A*B-(B**2)-(B**3))

def equileqnz(x):
    zl = x[0]
    zv = x[1]
    p = x[2]

    fugzl = fugz(tempp, zl, p)
    fugzv = fugz(tempp, zv, p)
    zcubicl = zcubic(tempp, zl, p)
    zcubicv = zcubic(tempp, zv, p)

    return [zcubicl, zcubicv, fugzl-fugzv]

tempp = 267

root_z = fsolve(equileqnz, [0.0002, 0.99, 0.041])

root = []
temp_list = [tempp]
root.append(root_z)
print(root[0])
```

    [2.52339063e-04 9.96564628e-01 4.47879861e-02]
    

동시에 풀어야 할 식이 3개이므로 3개의 식을 list형태로 return 해주는 `equileqnz`라는 함수를 정의한다. 3개의 식은 0이 되어야 하는 식(4), (5), (6)에 해당한다. 초기값은 식이 복잡하다보니 상당히 민감해서 거의 답 근처를 지정해 주어야 했다. 얻어진 답은 root라는 list에 담아준다. 이는 다른 온도에서의 답을 구하기 위한 초기값으로 사용될 것이다.


```python
for i in range(1, round(Tc)-267, 1):
    tempp = tempp + 1
    temp_list.append(tempp)
    root_z = fsolve(equileqnz, root[i-1])
    root.append(root_z)
    # print(f'temp = {tempp}, zl = {root_z[0]}, zv = {root_z[1]}, pres = {root_z[2]}')
```
온도를 1씩 증가시키면서 반복적으로 `fsolve` 해준다. 초기값은 직전step에서 구한 답을 활용한다.


```python
root_all = np.array(root)
temp_list = np.array(temp_list)

sat_pres = root_all[:,2]

denl = sat_pres/(root_all[:,0]*R*temp_list)
denv = sat_pres/(root_all[:,1]*R*temp_list)
```


[NIST Webbook](https://webbook.nist.gov/chemistry/fluid/)에서 아래와 같이 n-Hexane의 Saturated Properties를 얻을 수 있다.


|Temperature (K) | Pressure (bar) | Density (l, mol/l) | Density (v, mol/l) |
|:--:|--:|--:|--:|
|267|0.043069|7.9198|0.0019495|
|307|0.1215|7.7147|0.0051461|
|287|0.29223|7.504|0.011684|
|327|0.61954|7.2858|0.023603|
|347|1.1873|7.0582|0.043526|
|367|2.0971|6.8183|0.074768|
|387|3.4657|6.5624|0.12159|
|407|5.4239|6.2853|0.18976|
|427|8.1157|5.9785|0.28779|
|447|11.703|5.6277|0.42989|
|467|16.371|5.2031|0.64382|
|487|22.355|4.6283|1.0025|
|507|30.07|3.2679|2.151|

아래처럼 입력해준다. txt나 excel파일로 정리해서 python으로 읽어들어올수도 있다. 

```python
exp_temp = [267,287,307,327,347,367,387,407,427,447,467,487,507]
exp_denl = [7.9198,7.7147,7.504,7.2858,7.0582,6.8183,6.5624,6.2853,5.9785,5.6277,5.2031,4.6283,3.2679]
exp_denv = [0.0019495,0.0051461,0.011684,0.023603,0.043526,0.074768,0.12159,0.18976,0.28779,0.42989,0.64382,1.0025,2.151]
exp_psat = [0.043069,0.1215,0.29223,0.61954,1.1873,2.0971,3.4657,5.4239,8.1157,11.703,16.371,22.355,30.07]
exp_temp = np.array(exp_temp)
exp_denl = np.array(exp_denl)/1000
exp_denv = np.array(exp_denv)/1000
exp_psat = np.array(exp_psat)
```


```python
plt.plot(denl, temp_list)
plt.plot(denv, temp_list)
plt.scatter(exp_denl, exp_temp)
plt.scatter(exp_denv, exp_temp)
plt.xlabel('density (mol/cm3)')
plt.ylabel('Temperature (K)')
plt.title('saturated density vs temperature')
plt.show()

plt.plot(temp_list, sat_pres)
plt.scatter(exp_temp, exp_psat)
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (bar)')
plt.title('Saturated Vapor Pressure Curve')
plt.show()
```
    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-09-18/output_8_0.png)
    
    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-09-18/output_8_1.png)
    
데이터와 함께 이쁘게 Plot해주자. 계산값은 선에 해당하고 점들은 Data다. 계산이 잘 수행된 것 같다. Peng-Robinson 상태방정식은 꽤나 quality가 훌륭했다. 물리적으로 의미가 안맞을 수도 있지만 앞선 Regression 예제와 함께 생각해보면 데이터에 잘 맞는 파라미터($T_c, \ P_c, \ w$)를 얻을 수도 있을 것이다.

## Summary

비선형 방정식 풀이를 해봤다. `scipy.optimize.fsolve`를 활용했고 0이 되어야 할 함수를 지정해 주는 것과 초기값을 넣는 것에 주의해야 한다는 것을 알았다.

적당한 예시를 찾느라 상태방정식을 가져오긴 했는데, 지금와서 보니 그렇게 좋은 예시는 아니었던 것 같다. 식이 어떻게 저렇게 수립되는지는 전혀 신경 쓰지 말고 풀어야 하는 방정식 시스템이 어떻게 Python Code로 풀리는지 보는게 이번 포스팅의 의도와도 맞다.

어디까지나 Technical한 얘기였고 항상 마무리는 이거 다 필요없다는 거다. 정작 중요한 것은 풀어야 할 식을 수학적으로 표현해내는 과정이다. 여기가 실은 가장 시간을 많이 써야하는 부분이고 중요한 단계다.

