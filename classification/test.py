import cv2 
BLACK = [0,0,0]
 
img = cv2.imread('sample_image.jpg')

constant = cv2.copyMakeBorder(img,50,50,50,50,cv2.BORDER_REPLICATE)

cv2.imshow('pistol',constant)
cv2.waitKey(0)
cv2.imwrite('padded_image.jpg', constant)
