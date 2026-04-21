import cv2
import numpy as np

IMG_PATH = "debug/B_0_gray.png"
MOLD_SIZE = 150


def tophat_stage(img, mold_size):
    k_size = max(9, (mold_size // 8) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)


def otsu_thresh(img):
    _, th = cv2.threshold(img, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def get_contour_vis(binary):
    cnts, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, cnts, -1, (0,255,0), 1)
    return vis, len(cnts)


def main():
    gray = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print("Image not found")
        return

    filtered = cv2.bilateralFilter(gray, 9, 50, 50)
    th   = tophat_stage(filtered, MOLD_SIZE)
    otsu = otsu_thresh(th)
    vis, cnt = get_contour_vis(otsu)

    print(f"contours: {cnt}")

    label = np.zeros((20, vis.shape[1], 3), dtype=np.uint8)
    cv2.putText(label, f"bilateral+otsu  cnt={cnt}", (2,14),
                cv2.FONT_HERSHEY_PLAIN, 1, (200,200,200), 1)

    cv2.imshow("result", np.vstack([label, vis]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
