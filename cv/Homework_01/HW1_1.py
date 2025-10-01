import numpy as np
import cv2 as cv

def make_coords(w, h):
    return np.indices((w, h)).T

def bilinear_interpolation(origin_img, x_size, y_size):
    #출력 이미지 좌표 생성
    origin_h, origin_w = origin_img.shape[:2]
    O_coords = make_coords(x_size, y_size)
    #원본 이미지 좌표계로 변환
    M = np.array([[origin_w/x_size, 0], [0, origin_h/y_size]])
    O_coords = np.einsum('ij,...j->...i', M, O_coords)
    #p1,2,3,4의 좌표 구하기
    O_coords_x, O_coords_y = O_coords[..., 0], O_coords[..., 1]
    x0, y0 = np.floor(O_coords_x).astype(int), np.floor(O_coords_y).astype(int)
    x1, y1 = np.clip(x0 + 1, 0, origin_w - 1), np.clip(y0 + 1, 0, origin_h - 1)
    #p1,2,3,4 픽셀값 추출
    p1, p2, p3, p4 = origin_img[y0, x0], origin_img[y0, x1], origin_img[y1, x0], origin_img[y1, x1]
    
    P = np.stack([p1, p2, p3, p4], axis=-1)
    P = P.reshape(y_size, x_size, 2, 2)

    B = np.stack([1 - O_coords_y + y0, O_coords_y - y0], axis=-1)
    A = np.stack([1 - O_coords_x + x0, O_coords_x - x0], axis=-1)

    #바이리니어 공식 계산
    return np.einsum('...i,...ij,...j->...', B, P, A).astype(np.uint8)

def main():
    path, x_size, y_size = input("파일 경로 입력 : "), int(input("출력 가로 입력 : ")), int(input("출력 세로 입력 : "))

    origin_img = cv.imread(path)
    origin_img = cv.cvtColor(origin_img, cv.COLOR_BGR2GRAY)
    cv.imshow("origin_img", origin_img)
    
    interpolated_img = bilinear_interpolation(origin_img, x_size, y_size)
    cv.imshow(f'bilinear_interpolation to {x_size}x{y_size}', interpolated_img)

    print("회전된 이미지 출력 / q키 눌러 종료")
    
    while cv.waitKey(0) != ord('q'):
        pass
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()





