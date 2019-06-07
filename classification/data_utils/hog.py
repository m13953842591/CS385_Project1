import cv2
import os
import numpy as np
from config import *
from sklearn.preprocessing import normalize
bin_size = 9

def get_hog_cells(img, cs=16):
    """
    get hog features for each cell
    cs: cell size
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_size*ang/(2*np.pi))

    m, n = int(img.shape[0] / cs), int(img.shape[1]/cs)
    hog_cells = np.zeros((m, n, bin_size), dtype=np.float32)
    
    for i in range(m):
        for j in range(n):
            b = bins[i*cs:(i+1)*cs, j*cs:(j+1)*cs]
            m = mag[i*cs:(i+1)*cs, j*cs:(j+1)*cs]
            hog_cells[i][j] = np.bincount(b.ravel(), m.ravel(), bin_size)
            hog_cells[i][j] = hog_cells[i][j] / hog_cells[i][j].max()

    return hog_cells

def hog(img):
    """
    The input is [96, 96, 3] image
    The output is [900, ] hog feature
    """
    hog_cells = get_hog_cells(img)
    hists = np.zeros((25, 36), dtype=np.float32)
    for i in range(5):
        for j in range(5):
            hists[i*5+j] = np.hstack([hog_cells[i][j], hog_cells[i+1][j], hog_cells[i][j+1], hog_cells[i+1][j+1]])

    return np.hstack(hists)

def visualize(img, cs=16):
    """
    cs: cell size 
    """
    w = get_hog_cells(img, cs) * 255 / 4
    w = w.astype(np.uint8)
    bim0 = np.zeros((cs, cs), dtype=np.uint8)
    bim0[:, round(cs/2): round(cs/2)+1] = 1
    bim = np.zeros((cs, cs, bin_size), dtype=np.uint8)
    for i in range(0, bin_size):
        M = cv2.getRotationMatrix2D((cs/2, cs/2), -i*20, 1)
        bim[:, :, i] = cv2.warpAffine(bim0, M, (cs, cs))

    out_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            for k in range(bin_size):
                out_img[i*cs:(i+1)*cs, j*cs:(j+1)*cs] += bim[:, :, k] * w[i,j,k]
    
    cv2.imshow("hog feature visualize", out_img)
    print("out_img.shape = ", out_img.shape)
    cv2.waitKey(0)

if __name__ == "__main__":
    path = "2002/08/11/big/img_591.jpg"
    path = os.path.join(DATA_PATH + '/originalPics', path)
    img = cv2.imread(path)
    cv2.imshow("img", img)
    print("img.shape = ", img.shape)
    hog_feature = hog(img)
    visualize(img)