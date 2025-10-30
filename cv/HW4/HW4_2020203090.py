import numpy as np
import cv2 as cv

def rounding(arr):
    """
    .5 이상은 올리고, .5 미만은 내리는 반올림 적용
    혹시모를 오버플로를 방지하기 위해 [0, 255]로 클리핑
    uint8로 타입 변환후 반환
    """
    return np.clip(np.floor(arr + 0.5), 0, 255).astype(np.uint8)

def G_noise(src, m, s):
    img_f = src.astype(np.float32)
    h, w = src.shape[:2]

    # 1) 노이즈를 담을 비어있는 배열 생성
    noise_norm = np.empty_like(img_f, dtype=np.float32)

    if src.ndim == 3:
        # 각 채널을 순회
        for c in range(src.shape[2]):
            # cv.randn을 위한 연속적인 메모리 공간의 2D 임시 배열 생성
            channel_noise = np.empty((h, w), dtype=np.float32)
            
            # 임시 배열에 노이즈를 생성
            cv.randn(channel_noise, m, abs(s))
            
            # 생성된 노이즈를 원래 배열의 해당 채널에 복사
            noise_norm[:, :, c] = channel_noise
    else:
        # 흑백 이미지는 이미 연속적이므로 바로 적용
        cv.randn(noise_norm, m, abs(s))

    # 2) 이미지 도메인으로 역정규화(0~255)
    noise = noise_norm * 255.0

    noisy = img_f + noise
    return rounding(noisy)


def SnP_noise(src, p_s, p_p):
    # Salt(255) 확률 p_s, Pepper(0) 확률 p_p
    h, w = src.shape[:2]
    out = src.copy()
    r = np.random.rand(h, w)

    salt_mask   = r < p_s
    pepper_mask = (r >= p_s) & (r < p_s + p_p)

    if out.ndim == 3:
        out[salt_mask, :] = 255
        out[pepper_mask, :] = 0
    else:
        out[salt_mask] = 255
        out[pepper_mask] = 0
    return out

def pad_reflect(src, w):
    # 'reflect' 패딩(가장자리 값 제외 반사; OpenCV의 REFLECT_101과 유사)
    pad = w // 2
    if src.ndim == 2:
        return np.pad(src, ((pad, pad), (pad, pad)), mode='reflect')
    else:
        return np.pad(src, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

def median_filter(src, w):
    # 입력 src는 이미 pad_reflect가 적용된 상태
    pad = w // 2
    H, W = src.shape[:2]
    outH, outW = H - 2*pad, W - 2*pad

    if src.ndim == 2:
        src3 = src[..., None]
    else:
        src3 = src
    C = src3.shape[2]

    out = np.empty((outH, outW, C), dtype=np.float32)
    for i in range(outH):
        for j in range(outW):
            window = src3[i:i+w, j:j+w, :]  # (w, w, C)
            out[i, j, :] = np.median(window, axis=(0, 1))

    out = out.squeeze(-1) if C == 1 else out
    return rounding(out)

def mean_filter(src, w):
    # 입력 src는 이미 pad_reflect가 적용된 상태
    pad = w // 2
    H, W = src.shape[:2]
    outH, outW = H - 2*pad, W - 2*pad

    if src.ndim == 2:
        src3 = src[..., None]
    else:
        src3 = src
    C = src3.shape[2]

    out = np.empty((outH, outW, C), dtype=np.float32)
    inv_area = 1.0 / (w * w)
    for i in range(outH):
        for j in range(outW):
            window = src3[i:i+w, j:j+w, :].astype(np.float32)
            out[i, j, :] = window.sum(axis=(0, 1)) * inv_area

    out = out.squeeze(-1) if C == 1 else out
    return rounding(out)

def main():
    src = cv.imread("Lena512.jpg", cv.IMREAD_COLOR)

    cv.imshow("original", src)

    noise_img = {"Gaussian m=0 s=0.075": G_noise(src, 0, 0.075),
                 "Gaussian m=0 s=0.4" : G_noise(src, 0, 0.4),
                 "SnP p(s)=0.05 p(p)=0.05": SnP_noise(src, 0.05, 0.05),
                 "SnP p(s)=0.1 p(p)=0.1"  : SnP_noise(src, 0.1,  0.1)}
    
    using = {"mean_F"   : mean_filter,
             "median_F" : median_filter}
    
    win_size = 3

    for i_name, I in noise_img.items():
        cv.imshow(f'{i_name}', I)
        #cv.imwrite(f'{i_name}.png', I)
        for f_name, F in using.items():
            rst = F(pad_reflect(I, win_size), win_size)
            cv.imshow(f'{i_name}-{f_name}', rst)
            #cv.imwrite(f'{i_name}-{f_name}.png', rst)
            
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
