import cv2
import numpy as np 
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import math
import glob
import os
import matplotlib.pyplot as plt
import random
import time
import pandas as pd


# Store the Dataset as a list
path = 'stereo/centre/'
#f = [img for img in glob.glob(path)]
f=[]
for img in os.listdir(path):
	f.append(img)
	f.sort()
print(len(f))

# Extract Camera Parameters
fx, fy , cx , cy ,camera_image, LUT = ReadCameraModel('./model')

# Calibration Matrix
K =  np.array([[fx , 0 , cx], [0 , fy ,cy], [0 , 0 , 1]])

file1 = open("points.txt","a")

# SIFT to find Points of Correspondence

def findpts(img1,img2):

	# Initialization
	sift = cv2.xfeatures2d.SIFT_create()

	# Determine Key Points and Descriptors
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	# FLANN
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)

	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)

	# List to store features for two frames
	f1 = []
	f2 = []

	# Ratio Test
	X = []
	ratio = 0.5
	pts = []
	for i,(m,n) in enumerate(matches):
		if m.distance < ratio*n.distance:
			f1.append(kp1[m.queryIdx].pt)
			f2.append(kp2[m.trainIdx].pt)

	return f1,f2




	

# Main

# Preprocess Images
start_t = time.time()
H_init = np.identity(4)
p0 = np.array([[0,0,0,1]]).T
data = [] 
for i in range(19,25):
	print(i)
	img1 = cv2.imread(path+str(f[i]),0)
	# cv2.imshow("",img1)
	# cv2.waitKey(0)
	bgr_img1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
	undist_img1 = UndistortImage(bgr_img1,LUT)
	gray_img1 = cv2.cvtColor(undist_img1,cv2.COLOR_BGR2GRAY)


	img2 = cv2.imread(path+str(f[i+1]),0)
	bgr_img2 = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)
	undist_img2 = UndistortImage(bgr_img2,LUT)
	gray_img2 = cv2.cvtColor(undist_img2,cv2.COLOR_BGR2GRAY)

	gray_img1 = gray_img1[200:650 , 0:1280]
	gray_img2 = gray_img2[200:650 , 0:1280]

	# Find Points of Correspondence 

	f1,f2 = findpts(gray_img1,gray_img2)
	
	F ,m = cv2.findFundamentalMat
	