## 결과보고서 2020203090 한용옥
<br>

### 입력 이미지
`512 * 512` 의 컬러(`8UC3`) 레나 사진
<br>


|Average filter||
|:--:|:--:|
|![](./실행결과/P2%20Average%20Filter%203x3_screenshot_09.10.2025.png)|![](./실행결과/P2%20Average%20Filter%205x5_screenshot_09.10.2025.png)|
|`3*3`|`5*5`|
|![](./실행결과/P2%20Average%20Filter%2011x11_screenshot_09.10.2025.png)|![](./실행결과/P2%20Average%20Filter%2025x25_screenshot_09.10.2025.png)|
|`11*11`|`25*25`|

### Average filter
제로패딩을 사용하여 필터링 후 크기가 원본과 같게 하였다

커널의 크기가 커질수록 평균을 계산하는 영역이 넓어지므로 각 픽셀의 값이 주변 픽셀들의 영향을 더 많이 받게 된다 그 결과, 세부적인 픽셀 성분이 점차 사라지고 영상 전체가 부드럽고 흐릿하게 변한다

또한 본 실험에서는 제로 패딩을 사용하였기 때문에 커널 크기가 커질수록 영상의 경계부에서 `0`값을 포함하는 영역이 늘어나 가장자리 부분이 점점 더 어두워지며 검은 테두리처럼 나타난다

|Gaussian filter||
|:--:|:--:|
|![](./실행결과/P3%20GaussianBlur%203x3%20sigma=0.100000_screenshot_09.10.2025.png)|![](./실행결과/P3%20GaussianBlur%203x3%20sigma=1.000000_screenshot_09.10.2025.png)|
|`3*3` $\sigma=0.1$|`3*3` $\sigma=1$|
|![](./실행결과/P3%20GaussianBlur%203x3%20sigma=5.000000_screenshot_09.10.2025.png)|![](./실행결과/P3%20GaussianBlur%207x7%20sigma=0.100000_screenshot_09.10.2025.png)|
|`3*3` $\sigma=5$|`7*7` $\sigma=0.1$|
|![](./실행결과/P3%20GaussianBlur%207x7%20sigma=1.000000_screenshot_09.10.2025.png)|![](./실행결과/P3%20GaussianBlur%207x7%20sigma=5.000000_screenshot_09.10.2025.png)|
|`7*7` $\sigma=1$|`7*7` $\sigma=5$|

### Gaussian filter
가우시안 필터는 중심 픽셀 주변의 값을 가우시안 분포(정규분포) 형태의 가중치로 평균 내는 방식이다 따라서 $\sigma$ 는 분포의 폭을 결정하며, 필터 크기는 연산에 포함되는 영역의 범위를 결정한다  $\sigma$ 가 작을수록 중심 근처의 픽셀만 반영되어 약한 블러가 적용된다  $\sigma$ 가 커질수록 먼 픽셀까지 가중이 넓게 분포하여 부드럽고 흐릿한 영상이 된다 필터 크기(`3*3` → `7*7`) 를 키우면, 동일한  $\sigma$  값에서도 더 넓은 영역을 고려하기 때문에 연산 결과가 안정화되고 경계 부근에서의 노이즈가 완화된다

|문제 4,5,6,7,9 결과||
|:--:|:--:|
|![](./실행결과/P4%20Gaussian%20Noise%20(mean=0.000000,%20std=5.000000)_screenshot_09.10.2025.png)|![](./실행결과/P4%20Salt&Pepper%20Noise%20(salt=0.010000,%20pepper=0.010000)_screenshot_09.10.2025.png)|
|가우시안 노이즈 $\mu=0, \sigma=5$|SnP 노이즈 $p_{salt} = 0.01, p_{pepper} = 0.01$|
|![](./실행결과/P5%20Geometric%20Mean%203x3%20on%20Gaussian_screenshot_09.10.2025.png)|![](./실행결과/P5%20Geometric%20Mean%203x3%20on%20S&P_screenshot_09.10.2025.png)|
|가우시안 $\rightarrow$ 기하평균 `3*3`|SnP $\rightarrow$ 기하평균 `3*3`|
|![](./실행결과/P6%20Median%203x3%20on%20Gaussian_screenshot_09.10.2025.png)|![](./실행결과/P6%20Median%203x3%20on%20S&P_screenshot_09.10.2025.png)|
|가우시안 $\rightarrow$ 중앙값 `3*3`|SnP $\rightarrow$ 중앙값 `3*3`|
|![](./실행결과/P7%20Adaptive%20Median%20Smax=7%20on%20Gaussian_screenshot_09.10.2025.png)|![](./실행결과/P7%20Adaptive%20Median%20Smax=7%20on%20S&P_screenshot_09.10.2025.png)|
|가우시안 $\rightarrow$ Adaptive Median `Win_max=7`|SnP $\rightarrow$ Adaptive Median `Win_max=7`|
|![](./실행결과/P9%20Result.png)||
|모든 경우의 PSNR 수치||


