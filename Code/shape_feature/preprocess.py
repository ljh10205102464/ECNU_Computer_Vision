import cv2
import numpy as np

def preprocess(img):
    kernel_Ero = np.ones((3,1),np.uint8)
    kernel_Dia = np.ones((5,1),np.uint8)
    copy_img = img.copy()
    #copy_img = cv2.resize(copy_img,(400,600))
    cv2.imshow('copy_img',copy_img)
    cv2.waitKey(0)
    # 图像灰度化
    gray=cv2.cvtColor(copy_img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',gray)
    cv2.waitKey(0)
    # 高斯滤波
    imgblur=cv2.GaussianBlur(gray,(5,5),50)
    cv2.imshow('imgblur',imgblur)
    cv2.waitKey(0)
    #阈值处理
    ret,thresh=cv2.threshold(imgblur,140,255,cv2.THRESH_BINARY)
    cv2.imshow('thresh',thresh)
    cv2.waitKey(0)
    cv2.imwrite('thresh.png',thresh)
    #腐蚀
    img_Ero=cv2.erode(thresh,kernel_Ero,iterations=1)
    cv2.imshow('img_Ero',img_Ero)
    cv2.waitKey(0)
    #膨胀
    img_Dia=cv2.dilate(img_Ero,kernel_Dia,iterations=1)
    cv2.imshow('img_Dia',img_Dia)
    cv2.waitKey(0)

    return img_Dia