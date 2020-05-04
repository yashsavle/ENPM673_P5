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
#from numba import jit
#from numba import vectorize

# Store the Dataset as a list
path = 'stereo/centre/'

# Path to write frames
os.mkdir('timelapse/')
wrframe = 'timelapse/'

# Path to plot
os.mkdir('plots/')
pathplot = 'plots/'
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

# Open file to write camera coordinates
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


#@vectorize(['float32(float32, float32)'], target='cuda')
#@jit(nopython = True)

# Function to determine F
def findF(f1,f2):
	total = 0
	F_new = np.zeros((3,3))
	i1 = []
	i2 = []
	it = 10 # No. of iterations for RANSAC
	j=0
	for j in range(0,it):
		c = 0
		pts = []
		frame1 = []
		frame2 = []
		tmp1 = []
		tmp2 = []

        # 8 point Algorithm
		while True:
			n = random.randint(0,len(f1)-1)
			if n not in pts:
				pts.append(n)
			if len(pts) == 8:
				break

        # update features
		for p in pts:
			frame1.append([f1[p][0], f1[p][1]])
			frame2.append([f2[p][0], f2[p][1]])

		# Find Fundamental Matrix F
		F = run_ransac(frame1,frame2)

		for k in range(0, len(f1)):
			
			if checkval(f1[k], f2[k],F) < 0.01:
				c+=1
				#print('a\n')
				tmp1.append(f1[k])
				tmp2.append(f2[k])
		if c > total:
			total = c
			F_new = F
			i1 = tmp1
			i2 = tmp2
		
	return F_new,i1,i2

# Ransac to solve for F
def run_ransac(f1,f2):
	A = np.empty((8, 9))

	# Store 8 feature points in A matrix
	for i in range(0, len(f1)):
		x1 = f1[i][0]  
		y1 = f1[i][1] 
		x2 = f2[i][0] 
		y2 = f2[i][1] 
		A[i] = np.array([x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])

	# Take SVD to solve for F
	u, s, v = np.linalg.svd(A, full_matrices=True)  
	# Extract Solution 
	f = v[-1].reshape(3,3)

	# Remove noise by changing Rank to 2
	u1,s1,v1 = np.linalg.svd(f) 
	s2 = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]])
	F = u1 @ s2 @ v1  

	return F  


#@vectorize(['float32(float32, float32)'], target='cuda')
# 
def checkval(x1, x2 , F):
	a=np.array([x1[0],x1[1],1]).T
	b=np.array([x2[0],x2[1],1])
	return abs(np.squeeze(np.matmul((np.matmul(b,F)),a)))






# calculate the Essential Matrix with F and Camera Params
def findE(cal,F):
	t = np.matmul(np.matmul(cal.T,F),K)
	u,s,v = np.linalg.svd(t,full_matrices=True)
	sigmaF = np.array([[1,0,0],[0,1,0],[0,0,0]])
	a = np.matmul(u,sigmaF)
	E = np.matmul(a,v)
	return E


# @vectorize(['float32(float32, float32)'], target='cuda')
# Decompose E into T and R matrix
def decomp(E):
	u, s, v = np.linalg.svd(E, full_matrices=True)
	w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

	# Configuration 1
	c1 = u[:, 2] 
	r1 = u @ w @ v

	# Correct Camera Pose if determinant < 0
	if np.linalg.det(r1) < 0:
		c1 = -c1 
		r1 = -r1
	c1 = c1.reshape((3,1))

	# Configuration 2
	c2 = -u[:, 2]
	r2 = u @ w @ v
	if np.linalg.det(r2) < 0:
		c2 = -c2 
		r2 = -r2 
	c2 = c2.reshape((3,1))

	# Configuration 3
	c3 = u[:, 2]
	r3 = u @ w.T @ v
	if np.linalg.det(r3) < 0:
		c3 = -c3 
		r3 = -r3 
	c3 = c3.reshape((3,1)) 

	# Configuration 4
	c4 = -u[:, 2]
	r4 = u @ w.T @ v
	if np.linalg.det(r4) < 0:
		c4 = -c4 
		r4 = -r4 
	c4 = c4.reshape((3,1))

	return [r1, r2, r3, r4], [c1, c2, c3, c4]

#@vectorize(['float32(float32, float32)'], target='cuda')
# Estimate Depth of points
def cheirality_check(R, C, i1, i2):
	check = 0

	H = np.identity(4) 
	for i in range(0, len(R)): 

	    # find angles from R
		angles = get_orientation(R[i])


		if angles[0] < 50 and angles[0] > -50 and angles[2] < 50 and angles[2] > -50:
			count = 0 
			newP = np.hstack((R[i], C[i])) 
			for j in range(0, len(i1)): 
				X = linear_triangulation(H[0:3,:], newP, i1[j], i2[j]) 
				tr = R[i][2,:].reshape((1,3)) 
				if np.squeeze(tr @ (X - C[i])) > 0: 
					count = count + 1 

			if count > check: 
					check = count
					newC = C[i]
					newR = R[i]

	if newC[2] > 0:
		newC = -newC


	return newR, newC
#@vectorize(['float32(float32, float32)'], target='cuda')
# Find angles
def get_orientation(R):
	angle = math.sqrt(R[0,0]**2+ R[1,0]**2)
	flag = angle< 1e-6

	if not flag:
		x = math.atan2(R[2,1],R[2,2])
		y = math.atan2(-R[2,0],angle)
		z = math.atan2(R[1,0],R[0,0])

	else :
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], angle)
		z = 0
	return np.array([x*180/math.pi, y*180/math.pi, z*180/math.pi])

#@vectorize(['float32(float32, float32)'], target='cuda')
# Perform Linear Triangulation
def linear_triangulation(pose1,pose2,pt1,pt2):
	x_i = np.array([[0, -1, pt1[1]], [1, 0, -pt1[0]], [-pt1[1], pt1[0], 0]])
	x_i1= np.array([[0, -1, pt2[1]], [1, 0, -pt2[0]], [-pt2[1], pt2[0], 0]])
	A1 = x_i @ pose1[0:3, :] 
	A2 = x_i1@ pose2
	Ax = np.vstack((A1, A2))
	u, s, v = np.linalg.svd(Ax)
	new_X = v[-1]
	new_X = new_X/new_X[3]
	new_X = new_X.reshape((4,1))
	return new_X[0:3].reshape((3,1))   

#@vectorize(['float32(float32, float32)'], target='cuda')
# Homogeneous Matrix
def get_homogeneous(R, T):
	a = np.column_stack((R, T))
	b = np.array([0, 0, 0, 1])
	H = np.vstack((a, b))
	return H

# Main
def main():
	# Preprocess Images
	start_t = time.time()
	H_init = np.identity(4)
	p0 = np.array([[0,0,0,1]]).T
	data = [] 
	for i in range(18,len(f)-1):
		print(i)
		img1 = cv2.imread(path+str(f[i]),0)
		# cv2.imshow("",img1)
		# cv2.waitKey(0)
		# Frame 1
		bgr_img1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
		undist_img1 = UndistortImage(bgr_img1,LUT)
		gray_img1 = cv2.cvtColor(undist_img1,cv2.COLOR_BGR2GRAY)

		# Frame 2
		img2 = cv2.imread(path+str(f[i+1]),0)
		bgr_img2 = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)
		undist_img2 = UndistortImage(bgr_img2,LUT)
		gray_img2 = cv2.cvtColor(undist_img2,cv2.COLOR_BGR2GRAY)

		gray_img1 = gray_img1[250:600 , 0:1280]
		gray_img2 = gray_img2[250:600 , 0:1280]

		# Find Points of Correspondence 

		f1,f2 = findpts(gray_img1,gray_img2)

		for l in range(0,len(f1)):
			a = (int(f1[l][0]) , int(f1[l][1]))
			sift_img = cv2.circle(gray_img2, a,1,(255,0,0),3)
		cv2.imwrite(os.path.join(wrframe,str(i)+'.jpg'),bgr_img1)
		#sift_img = plt.imread(sift_img)
		# plt.imshow("",sift_img)
		# cv2.waitKey(0)
		# #print(f2)

		# Determine F

		F, i1 ,i2 = findF(f1,f2)
		
		# Determine E
		E = findE(K, F)

		#print(E)

		# Find T and R
		R_list, T_list = decomp(E)
		
		# Find Best estimates for R and T
		R , T = cheirality_check(R_list,T_list,i1,i2)

	    # Find Homogeneous Matrix and Camera Centre for Each frame

		H = get_homogeneous(R,T)
		H_init = H_init @ H
		p = H_init @ p0

		print('X = ',p[0])
		print('Y = ',p[2])

		# Write the Co-ordinates in a text file
		#data.append([p[0][0], -p[2][0]])
		file1.write(str(p[0][0])+",")
		file1.write(str(-p[2][0])+"\n")

		# Plot the Trajectory
		# plt.yticks(np.arange(-200,800,100))
		# plt.xticks(np.arange(0,1000,100))
		plt.scatter(p[0][0], -p[2][0], color='b')
		plt.savefig(os.path.join(pathplot,str(i)+'.png'))
	file1.close()
	end_t = time.time()
	delta = (end_t - start_t)/60
	print("Time taken: ",delta)
	plt.show()

if __name__ == '__main__':
	main()


