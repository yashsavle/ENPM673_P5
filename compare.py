import matplotlib.pyplot as plt 
import math

data1 = []
data2 = []
file1 = open("points.txt","r")
i=0
x1,y1 = zip(*[line.split(",") for line in file1])
#print (x[0])
#x,y = data1
file1.close()

file2 = open("points_predef.txt","r")
i=0
x2,y2 = zip(*[line2.split(",") for line2 in file2])
#print (x[0])
#x,y = data1
file1.close()
drift = 0
print(len(x2))
print(len(x1))
print(len(y1))
print(len(y2))
for i in range(0,len(x1)):
	drift += math.sqrt((float(x1[i])-float(x2[i]))**2+(float(y1[i])-float(y2[i]))**2)

avg = drift/len(x1)
	
print(drift)
print(avg)