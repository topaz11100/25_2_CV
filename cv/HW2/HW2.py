"""
이미지는 8UC1, 흑백 8bit로 간주
"""

import numpy as np
import cv2 as cv

def rounding(arr):
    """
    .5 이상은 올리고, .5 미만은 내리는 반올림 적용
    혹시모를 오버플로를 방지하기 위해 [0, 255]로 클리핑
    uint8로 타입 변환후 반환
    """
    return np.clip(np.floor(arr + 0.5), 0, 255).astype(np.uint8)

def gamma_transform(src, g):
    """
    감마 변환 함수
    T(p) = cp^{g}, c=1로 가정
    """
    result = 255 * np.power(src / 255, g)
    result = rounding(result)
    return result

def negative_transform(src):
    """
    반전 변환 함수
    """
    result = 255 - src
    return result

def hist_equalize(src):
    """
    히스토그램 균등화 구현
    이산형 공식 s_k = (L-1)\\sum_{j=0}^{k}{n_j} 구현
    """
    cum_count = np.cumsum(np.bincount(src.ravel()))
    matched_val = cum_count * 255 / (src.shape[0] * src.shape[1])
    matched_val = rounding(matched_val)
    result = matched_val[src]
    return result

def all_imshow(name_img_dict):
    """
    딕셔너리의 키 : 창 이름, 값 : 이미지
    모든 객체 띄우기
    """
    for k, v in name_img_dict.items():
        cv.imshow(k, v)

def main():
    """
    과제 1,2,3 진행
    """
    src = cv.imread("./Lena.png", cv.IMREAD_GRAYSCALE)
    
    G0_5, G1_5 = gamma_transform(src, 0.5), gamma_transform(src, 1.5)

    Ne = negative_transform(src)

    #과제 조건 : s = r/2,  s = 128 + r/2의 변환을 통해서 lena1, lena2영상을 만들기
    lena1, lena2 = rounding(src/2), rounding(128 + src/2)
    eq_src, eq_lena1, eq_lena2 = hist_equalize(src), hist_equalize(lena1), hist_equalize(lena2)

    window = {"Origin Img":src, "Gamma = 0.5":G0_5, "Gamma = 1.5":G1_5,
              "Negative transform":Ne, "Hist_eq_src":eq_src,
              "Hist_eq_lena1":eq_lena1, "Hist_eq_lena2":eq_lena2}
    all_imshow(window)

    cv.waitKey(0)
    cv.destroyAllWindows()

main()





