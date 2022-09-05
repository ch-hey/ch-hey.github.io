---
title: "[Eng. Calc.] Regression and Curve-Fit"
date: 2022-09-05 19:55:00 +0800
categories: [Engineering Calculation, Regression, Curve-Fit]
tags: [Engineering Calculation, Python, Regression, Curve-Fit, Density]
math: true
mermaid: true
---

이번 공학용 계산 주제는 Regression 이다. 편의상 Fitting 이라고도 한다. Regression, Fitting 원래는 명확한 구분이 있을거 같으면서도 굉장히 혼용되서 사용되는 단어 같다. 실은 딱히 명확한 구분은 없는 것 같다. 관련해선 [이 문서](https://stats.stackexchange.com/questions/151452/difference-between-regression-analysis-and-curve-fitting)를 한 번 읽어보자.

여기서는 변수, independent variable $x$에 대해서 함수값, 모델값, dependent variable $y$ 혹은 $f(x)$가 있을 때 이 둘 사이의 관계를 가장 잘 나타내주는 모델 (혹은 파라미터)를 찾는 것을 해보려고 한다. $x$, $y$는 모두 1-d, 2-d 혹은 n-d가 될 수 있다.

## 1. Intro

공학용 계산에서 Regression은 생각보다 자주 마주치는 (마주쳐야하는) 문제다. 어떤 데이터가 있고 이를 설명하는 이론식이 있다면 데이터에 잘 맞도록 하는 이론식 내 파라미터를 구하는 일들이다. 

굳이 얌전히 있는 데이터에 모델식을 끼워맞추는 일 같은건 왜하는 걸까. 하나는 비연속적인 데이터 사이를 메꿔주는 interpolation, 내삽의 의미가 있고 다른 하나는 데이터 외부를 예측해주는 extrapolation, 외삽, 예측, prediction의 의미가 있다. 

보통 좋은 모델이라고 불리는 이론식 일수록 예측의 범위와 신뢰도가 높고 모델식 내 Parameter가 물리적인 의미를 가지는 경우가 많다.

반대로 경험식, Empirical한 식의 경우 데이터 사이를 채워주는 정도로만 사용되며 조금만 범위를 넘어가면 예측의 신뢰도는 많이 떨어지고 물리적인 의미를 가지지 못하는 값을 예측하는 경우도 많다.

아래에는 두 가지 예제에 대해 Python으로 Regression 해보려고 한다. 

## 2. Regression - Mono-Variable

### 2.1. Saturated Liquid Density - Rackett Equation

포화 액체 밀도는 화공업계에서 상당히 많이 필요로 하는 물성 중에 하나다. 경험상 대부분 이 물성에 대해 크게 신경쓰지는 않지만 특수한 경우에는 정확한 예측이 중요할 수 있다. 대개 온도가 주어지면 포화액체 밀도를 구할 수 있는 이론식들이 사용된다.
(independent variable은 온도인 $T$이고 dependent variable은 포화액체밀도인 $d_{sat}$이다.)

포화 액체 밀도를 표현해 주는 식은 정말 다양하게 많다. 그 중에서 Rackett Equation을 사용하고자 한다. 공정모사 프로그램에서도 활용되는 식이다. 식 모양은 아래와 같이 간단한 형태다.

$$d_{sat} = d_c  B^{-(1-T/T_c)^N} \qquad (1)$$

식 1에서 $d_{sat}$은 우리가 알고자 하는 포화 액체 밀도이고 이를 위한 parameter로 B와 N이 있다. $d_c$는 critical liquid density, $T_c$는 critical temperature, $T$는 온도다.

아래 Python 코드를 통해 단순히 empirical한 모델식으로 polynomial 식을 사용해 보고 Rackett모델과 비교해 본다. Data는 [NIST Thermophysical Properties of Fluid Systems](https://webbook.nist.gov/chemistry/fluid/)를 사용하고 물질은  n-hexane이다.

### 2.2. Python Regression

사용할 모듈은 많지 않다. `numpy`, `scipy.optimize.curve_fit`, `scipy.optimize.minimize`가 regression에 쓰이고 `matplotlib`은 데이터 plot에 사용된다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
```


온도와 포화액체밀도 데이터다. 각각 Kelvin, mol/l 단위로 순서에 맞게 정리되어 있다.
알고보니 `numpy` 모듈에 `polyfit`이라는 기능이 있었다. 이 공식 [Document](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html) 보면 이거 하나로 n차 polynomial 식에 fitting 그냥 해준다. 


```python
Temp = [277.15, 287.15, 297.15, 307.15, 317.15, 327.15, 337.15, 347.15, 357.15, 367.15, 
        377.15, 387.15, 397.15, 407.15, 417.15, 427.15, 437.15, 447.15, 457.15, 467.15,
        477.15, 487.15, 497.15, 507.15, 507.82] # Temp in Kelvin(K)

Dens = [7.8164, 7.7132, 7.6086, 7.5024, 7.3943, 7.2842, 7.1716, 7.0564, 6.9382, 6.8164, 
        6.6907, 6.5604, 6.4249, 6.2831, 6.134, 5.9761, 5.8073, 5.6248, 5.4244, 5.1995, 
        4.9394, 4.6229, 4.1954, 3.2241, 2.706] # Liquid Density of Hexane in (mol/l)

Temp = np.array(Temp)
Dens = np.array(Dens)

Degree_of_polyfit = 2
coeff_np = np.polyfit(Temp, Dens, Degree_of_polyfit)
print(f"Coefficients for {Degree_of_polyfit}-poly fit are: {coeff_np}")
pred = np.poly1d(coeff_np)
print(f"AAD(%): {np.sum(((pred(Temp) - Dens)/Dens)**2)/len(Temp)*100:.3f}%")

```

    Coefficients for 2-poly fit are: [-7.77250536e-05  4.37154100e-02  1.41319326e+00]
    AAD(%): 0.575%
    

`np.polyfit`으로 온도와 밀도간 관계를 잘 설명해주는 2차 식의 계수를 구했고 상대 오차는 0.575%였다. 아래 그림처럼 적당한 Fit이 구해졌다.

```python
plt.title(f"{Degree_of_polyfit}-Poly fit Results")
plt.xlabel("Temperature (K)")
plt.ylabel("Molar Liquid Density (mol/l)")
plt.plot(Temp, pred(Temp))
plt.scatter(Temp, Dens)
plt.show()
```


    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-09-05/output_2_0.png)
    


이번에는 Rackett Equation을 써보려고 한다. 어떤 이론적 base를 가지는지는 잘 모르지만 어쨌든 식(1)은 polynomial 보다는 나을 것 같다. 아래 코드에서는 `curve_fit`기능을 사용해본다.

```python
tc = Temp[len(Temp)-1] #Tc = 507.82(K)
dc = Dens[len(Dens)-1] #dc = 2.706(mol/l)

def Rackett(temp, b, n):
    return dc*(b**(-(1-temp/tc)**n))

p0 = [0.1, 0.1]
para, conv = curve_fit(Rackett, Temp, Dens, p0)
print(f"Coefficients for Rackett Equation are: {para}")
predic = np.array(Rackett(Temp, *para))
print(f"AAD(%): {np.sum(((predic-Dens)/Dens)**2)/len(Temp)*100:.3f}%")
```

    Coefficients for Rackett Equation are: [0.26531865 0.28220359]
    AAD(%): 0.004%
    

식 (1)에서 $B$와 $N$이 각각 0.265, 0.282일 때 Rackett Equation이 n-hexane의 포화액체밀도를 잘 맞추었으며 상대오차는 0.004%였다.


```python
plt.title(f"Racektt Equation fit Results")
plt.xlabel("Temperature (K)")
plt.ylabel("Molar Liquid Density (mol/l)")
plt.plot(Temp, Rackett(Temp, *para))
plt.scatter(Temp, Dens)
plt.show()
```


    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-09-05/output_4_0.png)
    

앞서 기술했듯이 이론식의 장점은 예측범위와 신뢰도다. 어쩌면 당연한 얘기지만 2차 polynomial 식은 조금만 벗어나도 예측하는데 사용하지 못하는 수준이다. 

```python
Temp_expand = np.linspace(200, tc, 50)

plt.title(f"Fit Results")
plt.xlabel("Temperature (K)")
plt.ylabel("Molar Liquid Density (mol/l)")

plt.plot(Temp_expand, pred(Temp_expand), label = "Polyfit")
plt.plot(Temp_expand, Rackett(Temp_expand, *para), label = "Rackett")
plt.scatter(Temp, Dens, label = "Experiment")
plt.legend()
plt.show()
```


![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-09-05/output_5_0.png)    

아래는 `curve_fit`이 아닌 `minimize`를 쓰는 경우다. `minimize`를 사용하는 경우 cost function, objective function, 목적함수, residue, 편차 등으로 불리는 최소화 해줄 어떤 오차를 Racket_resid라고 정의해주었고 이를 최소화하는 방식으로 parameter를 얻었다. 

당연히 `curve_fit`을 쓰는 경우가 더 편하고 코드도 덜 쓰고 신경쓸 것도 적고 심지어 이 경우에 수렴도 조금 잘 되는 느낌을 받았다. 하지만 오차식을 정의하고 `minimize`를 쓰는 경우는 자유도가 높다. 할 수 있는게 있으니 분명히 언젠가 쓰인다. 

어쨌든 두 방식 모두 동일한 값을 얻었다.

```python
def Rackett_resid(p):
    b = p[0]
    n = p[1]
    res = 0
    for i in range(len(Temp)):
        res = res + ((Rackett(Temp[i], b, n) - Dens[i]))**2
    return res

p0 = [0.26, 0.28]
c = minimize(Rackett_resid, p0)
print(c.x)
```

    [0.26531862 0.28220363]
    


## 3. Regression - Multi-Variable

조금 더 복잡한 Regression으로 가보자. 위에서는 하나의 independent variable에 대해 Fitting 했다면 이번에는 2개다.

### 3.1. Polymer Density - Tait Equation

순수성분의 포화 액체 밀도라고 하면 온도만의 함수 혹은 압력만의 함수가 될 수 있다. 순수성분계에서 'Saturated', '포화'라는 단어가 이미 온도가 정해지면 압력은 정해지게끔 혹은 그 반대상황을 의미하기 때문이다. 여기서 포화라는 단어는 기체와 액체가 공존하는 상평형을 의미하기도 한다. 따라서 Rackett Equation은 온도 하나만의 식으로 표현될 수 있었다.

하지만 Polymer의 경우 '포화'라는 상태가 딱히 없다. 온도와 압력이 모두 주어져야 밀도가 정해진다.

이를 위한 이론식으로 [Tait Equation](https://polymerdatabase.com/polymer%20physics/EOS.html)이 있으며 아래와 같이 표현된다.

$$v(P,T) = v(0,T) \left( 1 - C \ln \left( 1 + {P \over B(T)}    \right)  \right) \qquad (2)$$

$$ B(T) = B_0\exp(-B_1T) \qquad (2a)$$

$$ v(0,T) = v_0\exp(-\alpha T) \qquad (2b)$$

$$ C = 0.0894 \qquad (2c)$$

여기서 $v(P,T)$는 임의의 압력, 온도에서 밀도의 역수인 specific volume을 나타낸다. $v(0,T)$는 압력이 0으로 고정되어 있고 임의의 온도 $T$일 때의 specific volume이다. 

식(2)에서 independent variable은 온도와 압력인 $T$와 $P$이며 dependent variable은 이때의 specific volume인 $v(T,P)$가 되며 모델식 내 parameter는 $B_0$, $B_1$, $v_0$, $\alpha$ 이다.

온도, 압력에 따른 specific volume 데이터를 활용하여 위 4개의 parameter들을 regression 할 것이다. 데이터는 P.Zoller의 [Standard Pressure-Volume-Temperature Data for Polymers](https://books.google.co.kr/books?id=e_LNtlIMlqEC&printsec=copyright&redir_esc=y#v=onepage&q&f=false)라는 책에서 일부 가져왔다. Polymer에 대한 정보는 아래와 같다.

- High Density Polyethylene
- Mw = 126,000 (PDI = 4.5)
- Specific Volume(at ambient condition) = 1.0537 cm3/g

### 3.2. Python Regression

먼저 data를 정리해보자. P.Zoller의 책에 데이터는 아래와 같이 테이블로 주어진다.


|Temp/Pressure | 0 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 |
|:--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
|170|1.3023|1.2781|1.2579|1.241|1.2264|1.2132|1.2015|1.1909|1.181|1.1718|1.1633|
|180|1.3124|1.2865|1.2654|1.2477|1.2326|1.219|1.2069|1.196|1.1859|1.1764|1.1677|
|189.7|1.3222|1.2951|1.2731|1.2546|1.2389|1.2249|1.2123|1.2011|1.1908|1.1812|1.1722|
|199.7|1.3327|1.3039|1.2809|1.2617|1.2455|1.2311|1.2182|1.2066|1.1961|1.1861|1.1772|
|209.7|1.3431|1.3124|1.2885|1.2687|1.2519|1.2371|1.2237|1.2118|1.2009|1.1908|1.1815|
|219.7|1.3535|1.3215|1.2963|1.2754|1.2582|1.2428|1.2292|1.2170|1.2058|1.1955|1.186|
|230.2|1.3639|1.3302|1.3041|1.2826|1.2645|1.2487|1.2345|1.222|1.2105|1.1999|1.1902|
|240.6|1.3755|1.3393|1.3121|1.2896|1.2711|1.2547|1.2403|1.2273|1.2156|1.2046|1.1946|
|249.9|1.3853|1.3476|1.319|1.296|1.2768|1.2598|1.2449|1.2318|1.2196|1.2085|1.1985|
|260.2|1.3964|1.3564|1.3268|1.3027|1.283|1.2656|1.2504|1.2368|1.2245|1.2133|1.2029|
|270.1|1.4075|1.3656|1.3344|1.3096|1.289|1.2712|1.2555|1.2417|1.229|1.2174|1.2068|

온도는 $^\circ C$, 압력은 Mpa단위다. 

위 table을 Python Code내에서 사용하고자 할 명령어가 알아들을 수 있게 정리해주는작업이 필요하다. 2차원 행렬에서 특정열 혹은 특정행을 지정하여 가져오는 일들은 python으로 어떤 일을 하려고 할 때 언젠가는 필수적으로 맞닥뜨릴 것이다. 자세한 설명은 하지 않지만 numpy array의 slicing, indexing을 검색해 보면 길을 찾을 수 있을 것이다.

여기서는 `curve_fit`만 사용할 예정이다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
```

아래 코드에서는 slicing을 통해 세로로 첫번째 열만을 지정할 수 있었다. v0T라는 변수에서 den이라는 2차배열을 어떻게 slicing하는지 이해하면 나중에 여러모로 유용하다.

```python
temp = [170, 180, 189.7, 199.7, 209.7, 219.7, 230.2, 240.6, 249.9, 260.2, 270.1]
temp = np.array(temp) + 273.15
pres = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
pres = np.array(pres)

den_170 = [1.3023, 1.2781, 1.2579, 1.241, 1.2264, 1.2132, 1.2015, 1.1909, 1.1810, 1.1718, 1.1633]
den_180 = [1.3124, 1.2865, 1.2654, 1.2477, 1.2326, 1.219, 1.2069, 1.196, 1.1859, 1.1764, 1.1677]
den_190 = [1.3222, 1.2951, 1.2731, 1.2546, 1.2389, 1.2249, 1.2123, 1.2011, 1.1908, 1.1812, 1.1722]
den_200 = [1.3327, 1.3039, 1.2809, 1.2617, 1.2455, 1.2311, 1.2182, 1.2066, 1.1961, 1.1861, 1.1772]
den_210 = [1.3431, 1.3124, 1.2885, 1.2687, 1.2519, 1.2371, 1.2237, 1.2118, 1.2009, 1.1908, 1.1815]
den_220 = [1.3535, 1.3215, 1.2963, 1.2754, 1.2582, 1.2428, 1.2292, 1.2170, 1.2058, 1.1955, 1.186]
den_230 = [1.3639, 1.3302, 1.3041, 1.2826, 1.2645, 1.2487, 1.2345, 1.222, 1.2105, 1.1999, 1.1902]
den_240 = [1.3755, 1.3393, 1.3121, 1.2896, 1.2711, 1.2547, 1.2403, 1.2273, 1.2156, 1.2046, 1.1946]
den_250 = [1.3853, 1.3476, 1.319, 1.296, 1.2768, 1.2598, 1.2449, 1.2318, 1.2196, 1.2085, 1.1985]
den_260 = [1.3964, 1.3564, 1.3268, 1.3027, 1.283, 1.2656, 1.2504, 1.2368, 1.2245, 1.2133, 1.2029]
den_270 = [1.4075, 1.3656, 1.3344, 1.3096, 1.289, 1.2712, 1.2555, 1.2417, 1.229, 1.2174, 1.2068]

den = np.array([den_170, den_180, den_190, den_200, den_210, den_220, den_230, den_240, den_250, den_260, den_270])

v0T = den[:,0]
print(v0T)
```

    [1.3023 1.3124 1.3222 1.3327 1.3431 1.3535 1.3639 1.3755 1.3853 1.3964
     1.4075]
    

먼저 식(2b)로 표현되는 압력이 0이고 임의의 온도 T에 대한 specific volume을 먼저 regression한다. 여기서는 온도만이 유일한 변수이다.


```python
plt.scatter(temp, v0T)

def v_at_temp0(temp, alpha, v0):
    
    return v0*np.exp(-alpha*temp)

p0 = [-0.0007, 0.9]
param, conv = curve_fit(v_at_temp0, temp, v0T, p0)

plt.plot(temp, v_at_temp0(temp, *param))
plt.scatter(temp, v0T)
plt.show()
```


    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-09-05/output_4_1.png)
    

다음으로 온도와 압력이 변수인 상태에서의 regression이다. 위에서 얻어진 (2b)에 대한 식은 그대로 활용한다.

아래 코드는 [이 문서](https://stackoverflow.com/questions/28372597/python-curve-fit-with-multiple-independent-variables)에서 코드를 참고했다.

```python
def tait(X, b0, b1):
    temp, pres = X
    C = 0.0894
    B = b0*np.exp(-b1*temp)
    return v_at_temp0(temp, *param)*(1-C*np.log(1+pres/B))

den_all = den.flatten()
pres_all = np.concatenate((pres, pres, pres, pres, pres, pres, pres, pres, pres, pres, pres))

temp_np = np.array([temp, temp, temp, temp, temp, temp, temp, temp, temp, temp, temp])

temp_all = np.concatenate((temp_np[:,0],temp_np[:,1],temp_np[:,2],temp_np[:,3],temp_np[:,4],
            temp_np[:,5],temp_np[:,6],temp_np[:,7],temp_np[:,8],temp_np[:,9],temp_np[:,10]))

para_all, conv = curve_fit(tait, (temp_all, pres_all), den_all)
print(para_all)
```

    [9.09419338e+02 5.29825619e-03]
    


```python
for i in range(len(pres)):
    if i % 2 == 0:
        plt.plot(temp, tait((temp, pres[i]),*para_all), label = f'{pres[i]} MPa')
        plt.scatter(temp, den[:,i])
    else:
        pass

plt.title('Polyethylene Specific Volume using Tait Equation')
plt.ylabel('Spec. Vol (cm3/g)')
plt.xlabel('Temp(K)')
plt.legend()
plt.show()
```


    
![png](https://raw.github.com/ch-hey/imgcdn/master/img/2022-09-05/output_6_1.png)
    



## Summary

공학용 계산 예제로 Regression 혹은 Fitting을 해봤다. 몇가지의 변수들로 이루어진 데이터가 있고 이들간의 관계를 어떻게든 수학적으로 표현하고자 하는 행동들이었다.

이론식이 있으면 이론식 내 파라미터를 데이터에 맞도록 하는 작업들이 수행되고 이론식이 없다면 아무식이나 갖다 써보면서 잘 맞는지 확인한다. 이런 식들을 모델이라고 하고 이런 일들을 모델링한다라고도 하는것 같다.

이러한 모델링은 결국 모델식과 데이터간의 오차를 최소화 하는 방식으로 이루어지며 이를 편하게 해주는 `curve_fit`과 `minimize`라는 기능이 `scipy.optimize` 모듈에 있었다. 물론 다른 모듈에도 비슷한 기능들이 많을 것이고 python 외에도 다른 선택지는 많다.

이런식의 모델링으로 하고자 하는 일은 결국 Case Study, 어떤 판단, 결국 예측이다. 하지만 항상 염두에 둬야 하는건 모든 모델링은 제한적으로만 맞고 대부분은 현실을 정확하게 표현해주지 못한다는 점이다. 이론식에 대한 배경, 이론을 알아가는 과정이 훨씬 더 중요한 이유다. 

이번 포스팅 내내 Regression하고 Fitting 하는 것만 잔뜩 써놨지만 정작 Fitting 하는 것 자체는 크게 어렵거나 중요하지 않다.