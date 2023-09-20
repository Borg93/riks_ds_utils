import cv2


class Preprocess():

    def binarize_img(img: str):

        img_ori = cv2.imread(img)
        img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
        dst = cv2.fastNlMeansDenoising(img_gray, h=31, templateWindowSize=7, searchWindowSize=21)
        img_blur = cv2.medianBlur(dst, 3).astype('uint8')
        threshed = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        return threshed
