import numpy as np
import cv2 as cv

def make_coords(w, h):
    return np.indices((w, h)).T

def rotate(origin_img, W, H, angle):
    result = make_coords(W, H)

    R = np.array([[ np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                  [ np.sin(np.radians(angle)),  np.cos(np.radians(angle))]])  
    result = np.einsum('ij,...j->...i', R, result)

    I = np.identity(2)
    C = 0.5 * np.array([W, H])
    C = (I - R) @ C
    result += C
    
    result = np.round(result).astype(int)
    x, y = result[..., 0], result[..., 1]
    valid = (0 <= x) & (x < W) & (0 <= y) & (y < H)

    result = np.where(valid, origin_img[np.clip(y, 0, H-1), np.clip(x, 0, W-1)], 0)
    return result.astype(np.uint8)

def main():
    path, x_size, y_size, angle = input("파일 경로 입력 : "), int(input("가로 크기 입력 : ")), int(input("세로 크기 입력 : ")), int(input("회전 각도 입력 : "))

    origin_img = cv.imread(path)
    origin_img = cv.cvtColor(origin_img, cv.COLOR_BGR2GRAY)
    cv.imshow("origin_img", origin_img)
    
    rotated_img = rotate(origin_img, x_size, y_size, angle)
    cv.imshow(f'rotate to {angle} degree', rotated_img)

    print("보간된 이미지 출력 / q키 눌러 종료")
    
    while cv.waitKey(0) != ord('q'):
        pass
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()





