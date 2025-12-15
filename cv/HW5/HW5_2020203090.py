import os
import numpy as np
import cv2 as cv

# -------------------- 공용 유틸 --------------------
def rounding(arr):
    """[0,255]로 클리핑 후 .5 반올림하여 uint8 변환"""
    return np.clip(np.floor(arr + 0.5), 0, 255).astype(np.uint8)

def save_img(output_dir, name, img):
    """
    화면 표시와 파일 저장을 수행
    output_dir: 결과 이미지를 저장할 폴더 경로
    name: 윈도우 타이틀/파일명 겸용 (예: 'no_smooth_abs_gx.png')
    """
    os.makedirs(output_dir, exist_ok=True)

    cv.imshow(name, img)
    save_path = os.path.join(output_dir, name)
    cv.imwrite(save_path, img)

# -------------------- 패딩 (Reflect-101, 1픽셀) --------------------
def pad_reflect101_1(src):
    """
    BORDER_REFLECT_101과 동일한 1픽셀 패딩을 넘파이로 직접 구현
    예시(1D): [a,b,c,d] -> pad1 -> [b,a | a,b,c,d | d,c]
    2D에서는 행/열에 대해 각각 위 규칙을 적용
    """
    if src.ndim != 2:
        raise ValueError("pad_reflect101_1 expects a 2D array (grayscale).")

    H, W = src.shape
    out = np.empty((H + 2, W + 2), dtype=src.dtype)

    # 중앙
    out[1:-1, 1:-1] = src

    # 좌우 가장자리 반사(가장자리 제외 반사)
    out[1:-1, 0]   = src[:, 1]     # 왼쪽
    out[1:-1, -1]  = src[:, -2]    # 오른쪽

    # 상하 가장자리 반사
    out[0, 1:-1]   = src[1, :]     # 위
    out[-1, 1:-1]  = src[-2, :]    # 아래

    # 네 모서리
    out[0, 0]      = src[1, 1]     # 좌상
    out[0, -1]     = src[1, -2]    # 우상
    out[-1, 0]     = src[-2, 1]    # 좌하
    out[-1, -1]    = src[-2, -2]   # 우하
    return out

# -------------------- 합성곱 (넘파이 구현) --------------------
def conv3x3_reflect101(img, kernel):
    """
    3x3 커널 합성곱. 패딩은 pad_reflect101_1 사용.
    img: uint8/float64 상관없음 (계산은 float64로)
    kernel: (3,3) numpy array
    """
    if img.ndim != 2:
        raise ValueError("conv3x3_reflect101 expects a 2D grayscale image.")
    if kernel.shape != (3, 3):
        raise ValueError("kernel must be 3x3.")

    # 패딩
    padded = pad_reflect101_1(img.astype(np.float64))

    # 슬라이딩 윈도우
    win = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    out = (win * kernel).sum(axis=(-2, -1))
    return out  # float64

# -------------------- Sobel (넘파이 구현) --------------------
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T.copy()

def sobel_gx_gy(img_gray):
    gx = conv3x3_reflect101(img_gray, SOBEL_X)
    gy = conv3x3_reflect101(img_gray, SOBEL_Y)
    return gx, gy

def normalize_to_u8(arr):
    """
    시각화를 위한 [0,255] 정규화 (순수 min-max 스케일링만)
    """
    a = arr.astype(np.float64)
    mn, mx = float(a.min()), float(a.max())
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    a = (a - mn) / (mx - mn) * 255.0
    return rounding(a)

# -------------------- Thresholding (상위 n% 유지) --------------------
def threshold_for_top_percent(edge_strength, top_percent):
    """
    edge_strength의 상위 top_percent% 픽셀만 남도록 하는 임계값을 계산.
    예: top_percent = 15.0 이면 상위 15%만 엣지로 남게 함.
    """
    if not (0.0 < top_percent < 100.0):
        raise ValueError("top_percent must be in (0, 100).")

    # 아래에서부터 (100 - top_percent)% 지점이 경계값
    cutoff_percentile = 100.0 - float(top_percent)
    T = float(np.percentile(edge_strength, cutoff_percentile))
    return T

def binarize(arr, thr):
    return (arr >= thr).astype(np.uint8) * 255

# -------------------- Noise --------------------
def add_gaussian_noise_randn(img_gray, sigma_frac):
    """
    Gaussian noise 추가 (cv.randn 사용)
    sigma_frac: 0~1 스케일에서 표준편차(1이면 255 std)
    """
    img = img_gray.astype(np.float32)
    noise = np.empty_like(img, dtype=np.float32)
    cv.randn(noise, 0, float(sigma_frac) * 255.0)
    out = img + noise
    return np.clip(out, 0, 255).astype(np.uint8)

# -------------------- Gaussian Smoothing --------------------
def gaussian_smooth_cv(img_gray, ksize, sigma):
    """
    OpenCV GaussianBlur 허용(과제안내문 명시)
    """
    k = int(ksize)
    if k % 2 == 0 or k <= 0:
        raise ValueError("ksize must be a positive odd integer.")
    return cv.GaussianBlur(img_gray, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))

# -------------------- 파이프라인 --------------------
def edge_pipeline(img_gray, output_dir, top_percent=15.0, save_prefix="base"):
    """
    1) Sobel gx, gy
    2) |gx|, |gy|, |gx|+|gy| (시각화용 정규화)
    3) |gx|+|gy|의 상위 top_percent%만 엣지로 남기도록 thresholding
    """
    gx, gy = sobel_gx_gy(img_gray)
    ax = np.abs(gx)
    ay = np.abs(gy)
    s  = ax + ay  # edge strength

    # 시각화용 정규화
    ax_u8 = normalize_to_u8(ax)
    ay_u8 = normalize_to_u8(ay)
    s_u8  = normalize_to_u8(s)

    # 상위 top_percent% 유지하도록 임계값 계산
    thr_val = threshold_for_top_percent(s, top_percent)
    edge_bin = binarize(s, thr_val)

    # 저장 및 디스플레이
    save_img(output_dir, f"{save_prefix}_abs_gx.png", ax_u8)
    save_img(output_dir, f"{save_prefix}_abs_gy.png", ay_u8)
    save_img(output_dir, f"{save_prefix}_abs_gx_plus_gy.png", s_u8)
    save_img(output_dir, f"{save_prefix}_edge_top_{int(top_percent)}.png", edge_bin)

    return thr_val  # 필요하면 디버깅용으로 사용 가능

def main():
    # ===================== 상수 ===========================
    IMAGE_PATH       = "/home/yongokhan/바탕화면/25_2_CV/source/Lena512.jpg"  # 입력 영상 경로
    OUTPUT_PATH      = "/home/yongokhan/바탕화면/25_2_CV/cv/HW5/실행결과"     # 출력 영상 경로
    KEEP_TOP_PERCENT = 9                           # 상위 몇 %를 엣지로 남길지
    NOISE_SIGMAS     = 0.25                        # 가우시안 노이즈 sigma 강도
    GAUSS_CFGS       = [(3, 1), (9, 2)]         # (커널크기, 시그마) 목록
    # ======================================================

    # 0) 입력
    src = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

    # 1) 원본에 대한 엣지
    edge_pipeline(
        src,
        OUTPUT_PATH,
        top_percent=KEEP_TOP_PERCENT,
        save_prefix="src"
    )

    # 2) 노이즈 추가 후 동일 파이프라인
    noisy = add_gaussian_noise_randn(src, NOISE_SIGMAS)
    save_img(OUTPUT_PATH, f"noisy_sigma_{NOISE_SIGMAS}.png", noisy)
    edge_pipeline(
        noisy,
        OUTPUT_PATH,
        top_percent=KEEP_TOP_PERCENT,
        save_prefix=f"noisy_sigma_{NOISE_SIGMAS}"
    )

    # 3) Gaussian smoothing 후 (각 이미지에서 상위 KEEP_TOP_PERCENT% 엣지 유지)
    for (k, sig) in GAUSS_CFGS:
        sm = gaussian_smooth_cv(noisy, k, sig)
        save_img(OUTPUT_PATH, f"gauss_{k}x{sig}.png", sm)
        edge_pipeline(
            sm,
            OUTPUT_PATH,
            top_percent=KEEP_TOP_PERCENT,
            save_prefix=f"gauss_{k}x{sig}_edge"
        )

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
