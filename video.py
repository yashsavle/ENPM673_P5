import cv2
import numpy as np
import glob
import os

font = cv2.FONT_HERSHEY_SIMPLEX
origin = (50,50)
fontscale = 1
fontColor = (0,0,255)
thickenss = 2

fourcc =cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,20,(960,720))


name=18
for i in range(0,3853):

	path2 = 'plots/'+str(name)+'.png'
	userdef = cv2.imread(path2)
	userdef = cv2.putText(userdef,'Userdefined',origin,font,fontscale,fontColor,thickenss)

	name+=1
	path1 = 'timelapse_p/'+str(name)+'.jpg'
	tl = cv2.imread(path1)
	
	path3 = 'plots_p/'+str(name)+'.png'
	predef = cv2.imread(path3)
	predef = cv2.putText(predef,'Predefined',origin,font,fontscale,fontColor,thickenss)

	stack1 = np.concatenate((predef,userdef),axis = 1)

	stack2 = np.concatenate((tl,stack1) , axis = 0)
	stack2 = cv2.resize(stack2,(960,720))
	# cv2.imshow("",stack2)

	out.write(stack2)
	print(name)
	# if cv2.waitKey(60) & 0x0FF ==ord('q'):
	# 	break 

out.release()
cv2.destroyAllWindows()
