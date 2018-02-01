import cv2
import numpy as np

def histogramEqualiseColour(img):
    # # set
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # split into channels
    # channels=cv2.split(ycrcb)
    # equalise
    # cv2.equalizeHist(channels[0],channels[0])
    # apply contrast limiting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    # merge all the channels
    # cv2.merge(channels, ycrcb)
    # convert back to ycrcb
    # cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,clahe_img)

    return img


photo_name = './histogram_test_images/orange4.jpg'
img = cv2.imread(photo_name, 1)
img = histogramEqualiseColour(img)
# cv2.imshow("ypp", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('img2.jpg', img)