import cv2
import numpy as np 
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import math


# Store the Dataset as a list
path = 'stereo/centre/'
f = [img for img in glob.glob(path)]
f.sort()


# Extract Camera Parameters
fx, fy , cx , cy , G camera image, LUT = ReadCameraModel('./model')

# Calibration Matrix
K =  np.array([[fx , 0 , cx], [0 , fy ,cy], [0 , 0 , 1]])



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

    # Filter Accurate Matches
    match_filter = [[(0,0) for i in range(len(matches))]]

    # Ratio Test
    X = []
    ratio = 0.5
    pts = []
    for i,(m,n) in enumerate(matches):
        if m.distance < ratio*n.distance:
            match_filter[i]=[1,0]
            X.append(m)
            pts.append([kp1[m.queryIdx].pt, kp2[m.trainIdx].pt]) 
    pts = np.array(pts)

    return pts


def fundamental_mat(gf1,gf2):
    A = np.empty((8, 9))

    for i in range(0, len(corners1)): # Looping over all the 8-points (features)
        x1 = corners1[i][0] # x-coordinate from current frame 
        y1 = corners1[i][1] # y-coordinate from current frame
        x2 = corners2[i][0] # x-coordinate from next frame
        y2 = corners2[i][1] # y-coordinate from next frame
        A[i] = np.array([x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])

    u, s, v = np.linalg.svd(A, full_matrices=True)  # Taking SVD of the matrix
    f = v[-1].reshape(3,3) # Last column of V matrix
    
    u1,s1,v1 = np.linalg.svd(f) 
    s2 = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]]) # Constraining Fundamental Matrix to Rank 2
    F = u1 @ s2 @ v1  
    
    return F  



def checkval(x1, x2 , F):
    x11=np.array([x1[0],x1[1],1]).T
    x22=np.array([x2[0],x2[1],1])
    return abs(np.squeeze(np.matmul((np.matmul(x22,F)),x11)))


def use_ransac(pts):

	numpts = pts.shape[0]
	p_axis = np.ones((numpts,1))

	pts1 = np.concatenate((pts[:,0,:],p_axis),axis = 1)
	pts2 = np.concatenate((pts[:,1,:],p_axis),axis = 1)

	n_inliers = 0
	final_F = np.zeros((3,3))
		
	iter = 50
	i1 = []
	i2 = []

	while iter >0:
		count = 0
		eightpoint = []
		good_features1 = []
		good_features2 = []
		tmp1 = []
		tmp2 = []

		while(True):
            num = random.randint(0, len(pts1)-1)
            if num not in eightpoint:
                eightpoint.append(num)
            if len(eightpoint) == 8:
                break


        for point in eightpoint: # Looping over eight random points
            good_features1.append([pts1[point][0], pts1[point][1]]) 
            good_features2.append([pts2[point][0], pts2[point][1]])


        # Fundamental Matrix
        F = fundamental_mat(good_features1,good_features2)

        for n in range(0,len(pts1)):

        	if checkval(good_features1[n], good_features2[n],F)<0.01:
        		count = count + 1
        		tmp1.append(pts1[n])
        		tmp2.append(pts2[n])


        	if count > n_inliers:
        		n_inliers = count
        		final_F = F
        		i1 = tmp1 
        		i2 = tmp2
    return final_F,i1,i2




def findE(K,F):
	t = np.matmul(np.matmul(K.T,F),K)
	u,s,v = np.linalg.svd(t,full_matrices=True)
	sigmaF = np.array([[1,0,0],[0,1,0],[0,0,0]])
	a = np.matmul(u,sigmaF)
	E = np.matmul(a,v)
	return E




def findposeE(essentialMatrix):
	u, s, v = np.linalg.svd(essentialMatrix, full_matrices=True)
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # 1st Solution
    c1 = u[:, 2] 
    r1 = u @ w @ v
    
    if np.linalg.det(r1) < 0:
        c1 = -c1 
        r1 = -r1
    c1 = c1.reshape((3,1))
    
    # 2nd Solution
    c2 = -u[:, 2]
    r2 = u @ w @ v
    if np.linalg.det(r2) < 0:
        c2 = -c2 
        r2 = -r2 
    c2 = c2.reshape((3,1))
    
    # 3rd Solution
    c3 = u[:, 2]
    r3 = u @ w.T @ v
    if np.linalg.det(r3) < 0:
        c3 = -c3 
        r3 = -r3 
    c3 = c3.reshape((3,1)) 
    
    # 4th Solution
    c4 = -u[:, 2]
    r4 = u @ w.T @ v
    if np.linalg.det(r4) < 0:
        c4 = -c4 
        r4 = -r4 
    c4 = c4.reshape((3,1))
    
    return [r1, r2, r3, r4], [c1, c2, c3, c4]


def bestRT(Rlist, Clist, features1, features2):
    check = 0
    Horigin = np.identity(4) # current camera pose is always considered as an identity matrix
    for index in range(0, len(Rlist)): # Looping over all the rotation matrices
        angles = rotationMatrixToEulerAngles(Rlist[index]) # Determining the angles of the rotation matrix
        #print('angle', angles)
        
        # If the rotation of x and z axis are within the -50 to 50 degrees then it is considered down in the pipeline 
        if angles[0] < 50 and angles[0] > -50 and angles[2] < 50 and angles[2] > -50: 
            count = 0 
            newP = np.hstack((Rlist[index], Clist[index])) # New camera Pose 
            for i in range(0, len(features1)): # Looping over all the inliers
                temp1x = getTriangulationPoint(Horigin[0:3,:], newP, features1[i], features2[i]) # Triangulating all the inliers
                thirdrow = Rlist[index][2,:].reshape((1,3)) 
                if np.squeeze(thirdrow @ (temp1x - Clist[index])) > 0: # If the depth of the triangulated point is positive
                    count = count + 1 

            if count > check: 
                check = count
                mainc = Clist[index]
                mainr = Rlist[index]
                
    if mainc[2] > 0:
        mainc = -mainc
                
    #print('mainangle', rotationMatrixToEulerAngles(mainr))
    return mainr, mainc

# Main

# Preprocess Images

for i in range(0,len(f)-1):

	img1 = cv2.imread(f[i])
	bgr_img1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
	undist_img1 = UndistortImage(img1,LUT)
	gray_img1 = cv2.cvtColor(undist_img1,cv2.COLOR_BGR2GRAY)


	img2 = cv2.imread(f[i+1])
	bgr_img2 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
	undist_img2 = UndistortImage(img1,LUT)
	gray_img2 = cv2.cvtColor(undist_img1,cv2.COLOR_BGR2GRAY)


	# Find Points of Correspondence 

	pts = findpts(gray_img1,gray_img2)


	# Estimate  Fundamental Matrix using RANSAC

	F, i1 ,i2 = use_ransac(pts)
	E = findE(K, F)

	# Find T and R
	R_list, T_list = findposeE(E)
	R , T = bestRT(R_list,T_list,i1,i2)


