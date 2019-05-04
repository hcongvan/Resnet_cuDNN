import cv2
import numpy as np
f = open('E:/Youtube/sample_per_title/frames0.yuv','rb')

Y= np.array([0])
y = f.read(1280*720)
uv = f.readline()
for i in y:
    Y = np.append(Y,i)
