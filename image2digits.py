# coding:utf8

import sys
import pytesseract
import cv2
import caffe
import numpy as np
import scipy.signal as signal

MEAN = 128
SCALE = 0.00390625
model5 = './3x3_mnist_usps_iter_200000.caffemodel'     
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net('classification_3x3.prototxt', model5, caffe.TEST)
net.blobs['data'].reshape(1, 1, 28, 28)

imglist = sys.argv[1]
img0 = cv2.imread(imglist)
h, w, _ = img0.shape
img = cv2.imread(imglist, cv2.IMREAD_GRAYSCALE)

d = pytesseract.image_to_boxes(img, lang="chi_sim", config="-c tessedit_char_blacklist=\\'′~～  -psm 3 -oem 3")
#  、- ' ~ as .
print(d)
size1, size2 = 0, 0
for b in d.splitlines():
    digit=''
    b = b.split(' ')
    pad = int((int(b[3])-int(b[1]))/5)
    x0, y1, x1, y0 = int(b[1]) - pad, h - int(b[2]) + pad, int(b[3]) + pad, h - int(b[4]) - pad
    cv2.rectangle(img0, (x0, y0), (x1, y1), (0, 255, 0), 2)
    size2 = y1 - y0

    #if b[0]=='、' or b[0]=='-' or b[0]=="'" or b[0]=='‘' or b[0]=='~' or b[0]=='′':
    if size1 > 0 and size2 < size1/2:    #check '.'
        digit='.'
        cv2.putText(img0, digit, (x0, y0), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        size1=size2
        continue
    size1=size2
    
    image = img[y0:y1,x0:x1]    # for handwritten digits recognition
    #print(image.shape)
    height,width =image.shape
    if height==0 or width==0:    # non-digits
        continue
    
    thresh, img_bin = cv2.threshold(image, 127, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image = 255-img_bin
       
    if height>width:    # pad to square
        p=int((height-width)/2)
        image=cv2.copyMakeBorder(image, 0, 0, p, p, cv2.BORDER_CONSTANT,0)
    elif height<width:
        p=int((width-height)/2)
        image=cv2.copyMakeBorder(image, p, p, 0, 0, cv2.BORDER_CONSTANT,0)
    
    kernal = np.ones((3,3))
    image_array = np.array(image)
    image_blur = signal.convolve2d(image_array,kernal,mode="same")
    image2 = (image_blur/float(image_blur.max()))*255
    image2[image2 >= image2.mean()] = 255
    image2[image2 <  image2.mean()] = 0
    image=cv2.resize(image2,(28,28))    # resize to 28x28

    image=image.astype(np.float) - MEAN
    image *= SCALE
    net.blobs['data'].data[...] = image
    output = net.forward()  # recognition
    digit = np.argmax(output['prob'][0])
    #print(digit)
    cv2.putText(img0, str(digit), (x0, y0), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

cv2.imshow('img', img0)
cv2.waitKey(0)

