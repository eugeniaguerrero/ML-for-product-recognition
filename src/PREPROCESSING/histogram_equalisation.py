import cv2
import numpy as np

def CLAHE_equalisation(img):
    img = np.array(img)
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)

    for i in range(img.shape[0]):
        # # set
        lab = cv2.cvtColor(img[i], cv2.COLOR_BGR2LAB)
        # split into channels
        l, a, b, = cv2.split(lab)
        # apply contrast limiting
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # merge enhanced L-channel with a and b channels
        l_img = cv2.merge((cl,a,b))
        # convert back to RGB model
        final = cv2.cvtColor(l_img, cv2.COLOR_LAB2BGR)

        img[i] = final

    return img


# photo_name = './orange2.jpg'
# img = cv2.imread(photo_name, 1)
# processed = CLAHE_equalisation(img)
# # cv2.imshow("ypp", img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# cv2.imwrite('processed2.jpg', processed)