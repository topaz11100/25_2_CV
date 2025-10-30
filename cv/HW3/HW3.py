import numpy as np
import cv2 as cv

def rounding(arr):
    """
    .5 이상은 올리고, .5 미만은 내리는 반올림 적용
    혹시모를 오버플로를 방지하기 위해 [0, 255]로 클리핑
    uint8로 타입 변환후 반환
    """
    return np.clip(np.floor(arr + 0.5), 0, 255).astype(np.uint8)

def make_avg_kernel(n):
    """
    이동 평균 커널 반환 함수
    커널은 정사각형이며 한 변의 길이 = 2n+1
    """
    size, value = (2*n + 1, 2*n + 1), 1 / (2*n + 1) ** 2
    return np.full(size, value, dtype=np.float64)

def laplacian():
    """
    라플라시안 정의에 따른 커널
    이미지에 커널 합성곱 시 값은 라플라시안이 된다
    """
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

def sharpen_based_laplacian(n):
    """
    이미지에 라플라시안을 빼면 픽셀 경계차가 더 심해져
    인간이 인지하기에 경계가 더 선명히 보인다
    그러한 과정은 f-h*f = (1-h)*f 이고
    본 함수는 (1-h)를 반환한다
    """
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

def zero_padding(src, n):
    """
    Zero Padding (cv2.BORDER_CONSTANT)
    n : (필터 크기 - 1) // 2
    """
    size = (src.shape[0] + 2*n, src.shape[1] + 2*n) + src.shape[2:]
    out = np.zeros(size, dtype=src.dtype)
    out[n:n+src.shape[0], n:n+src.shape[1], ...] = src
    return out

def apply_kernel(src, kernel):
    """
    입력이미지에 대해 커널 합성곱된 이미지 반환
    패딩 등 전처리는 되어있다 가정
    """
    win = np.lib.stride_tricks.sliding_window_view(src, window_shape=kernel.shape, axis=(0, 1))
    out = (win * kernel).sum(axis=(-2, -1))
    return out

def filtering(src, n, kernel):
    """
    종합적용, 원본, 필터크기, 커널, 패딩방식
    입력받아 필터링된 이미지를 반환
    """
    kernel = kernel(n)
    out = zero_padding(src, n)
    out = apply_kernel(out, kernel)
    out = rounding(out)
    return out

def display_laplacian(src):
    def laplacian_normalize(arr):
        """
        라플라시안 값 [0,255] 범위로 정규화.
        """
        arr_min, arr_max = np.min(arr), np.max(arr)

        # 모든 값이 동일할 경우(분모=0) 대비 처리
        if arr_max == arr_min:
            return np.zeros_like(arr, dtype=np.uint8)

        norm = (arr - arr_min) / (arr_max - arr_min)  # 0~1 사이로 스케일링
        norm *= 255              # [0,255]로 변환
        return rounding(norm)                 # 이미지 타입으로 변환

    out = zero_padding(src, 1)
    out = apply_kernel(out, laplacian())
    return laplacian_normalize(out)

def all_imshow(name_img_dict):
    """
    딕셔너리의 키 : 창 이름, 값 : 이미지
    모든 객체 띄우기
    """
    for k, v in name_img_dict.items():
        cv.imshow(k, v)

def main():
    """
    과제 메인 함수
    """
    source_path = input("Image path : ")

    color_mode = input("color mode ([color], [gray]) : ")
    
    if color_mode == "color":
        src = cv.imread(source_path, cv.IMREAD_COLOR)
    else:
        src = cv.imread(source_path, cv.IMREAD_GRAYSCALE)

    result = {"original": src,
              "avg_3":  filtering(src, 1, make_avg_kernel),
              "avg_5":  filtering(src, 2, make_avg_kernel),
              "avg_7":  filtering(src, 3, make_avg_kernel),
              "avg_11": filtering(src, 5, make_avg_kernel),
              "sharpen": filtering(src, 1, sharpen_based_laplacian)}
    
    lap = display_laplacian(src)
    if color_mode == "gray":
        result["Laplacian"] = lap
    elif color_mode == "color":
        L_b, L_g, L_r = lap[:, :, 0], lap[:, :, 1], lap[:, :, 2]
        result.update({"B_laplacian": L_b, "G_laplacian": L_g, "R_laplacian": L_r})
    
    all_imshow(result)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__=="__main__":
    main()





