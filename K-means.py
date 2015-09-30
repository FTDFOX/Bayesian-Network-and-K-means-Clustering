from numpy import *
from numpy.random import *
import numpy as np
import random
import json
import math

#globle parameters:
k=10
threshold=0.0001
V=[[] for i in range(k)]
numoftrainpoints=500
numoftestpoints=120
#X=[[] for i in range(10)]
#Y=[[] for i in range(10)]
X=[]
Y=[]
points=[]
testpoints=[]
trainlabels=[] #class
testlabels=[]
assignment=[[] for i in range(k)]# record witch point belong to which seed
map_v_given_c=[[] for i in range(6)]# 6 is the how many labels
difference=999999
num_iteration=0


#read file
with open("training.json",'r') as f:
	data= json.load(f)
	trainlabels=data["labels"]
	points=data["points"]
	for i in range(numoftrainpoints):
		X.append(points[i][0])
		Y.append(points[i][1])
	#for i in range(numoftrainpoints):
		#X.append(data["points"][i][0])
		#Y.append(data["points"][i][1])
with open("testing.json","r") as f:
	data=json.load(f)
	testlabels=data["labels"]
	testpoints=data["points"]
#f= open("training.txt","r")
#for line in f:
#	trainlabels.append(line.split("/n")[-1])
#trainlabels

#init V
def generate_init_vector():
	for v in range(k):
		x=random.uniform(min(X),max(X))
		y=random.uniform(min(Y),max(Y))
		V[v]=(x,y)

generate_init_vector()
V=list(V)
print(V)

#K means alg
def distance(point_1,point_2):
	#a=pow((point_1[0]-point_2[0]),2)
	#b=pow((point_1[1]-point_2[1]),2)
	return (point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1])

#testkey=[]


#for v in range(k):
#	del assignment[v][0]


# get X Y
def getXY():
	global difference
	difference=0
	for v in range(k):
		l=len(assignment[v])
		#temp_x=[]
		#temp_y=[]
		sum_x=0
		sum_y=0
		difference_=0
		for i in range(l):
			sum_x=sum_x+assignment[v][i][0]
			sum_y=sum_y+assignment[v][i][1]
		if l==0:
			V[v]=V[v]
		else:
			difference_=difference_+sqrt(pow(sum_x/l-V[v][0],2)+pow(sum_y/l-V[v][1],2))
			V[v]=(sum_x/l,sum_y/l)
		difference=difference+difference_


			#temp_y=[]
			#temp_x.append(assignment[v][i][0])
			#temp_y.append(assignment[v][i][1])
		#X.append(temp_x)
		#Y.append(temp_y)
##################################### remember to reset X and Y@#################

#def initi():
#	generate_init_vector()
#	V=list(V)


def k_means():
	#initi()
	global assignment
	global map_v_given_c
	while(difference>threshold):
		k_means_1st()
		getXY()
		print(difference)
		print(V)
		assignment=[[] for i in range(k)]
		map_v_given_c=[[] for i in range(6)]
	else:
		k_means_1st()
		getXY()
		print(difference)
		print(V)

        #M-step: update V
        #for v in range(k):
        #	#V=list(V)
        #	if X[v] and Y[v]:
        #			V[v]=(np.mean(X[v]),np.mean(Y[v]))
        #	print(V)
        	#V[v][0]=np.mean(X[v])
        	#V[v][1]=np.mean(Y[v])
        	#V=tuple(V)
#k_means()

def k_means_1st():
	#init()
	global num_iteration
	num_iteration=num_iteration+1
	for i in range(numoftrainpoints):
        	key=-1
        	shortest=9999999999.0
        	for v in range(0,k):
        		#key=-1
        		distance_=distance(V[v],points[i])
        		if distance_<shortest:
        			shortest=distance_
        			key=v
        	#testkey.append(key)		
        	assignment[key].append(points[i])
        	map_v_given_c[trainlabels[i]].append(key)



k_means()
print num_iteration
###################################################################


#Step 2
# compute P(c) 
#global parameters
distributionoflabels=[]
count0=0.0
count1=0.0
count2=0.0
count3=0.0
count4=0.0
count5=0.0
pointoflabel0=[]
pointoflabel1=[]
pointoflabel2=[]
pointoflabel3=[]
pointoflabel4=[]
pointoflabel5=[]

for i in range(numoftrainpoints):
	if trainlabels[i]==0:
		pointoflabel0.append(i)
		count0=count0+1
	else: 
		if trainlabels[i]==1:
			pointoflabel1.append(i)
			count1=count1+1
		else: 
			if trainlabels[i]==2:
				pointoflabel2.append(i)
				count2=count2+1
			else: 
				if trainlabels[i]==3:
					pointoflabel3.append(i)
					count3=count3+1
				else:
					if trainlabels[i]==4:
						pointoflabel4.append(i)
						count4=count4+1
					else: 
						pointoflabel5.append(i)
						count5=count5+1

distributionoflabels.append(count0/numoftrainpoints)
distributionoflabels.append(count1/numoftrainpoints)
distributionoflabels.append(count2/numoftrainpoints)
distributionoflabels.append(count3/numoftrainpoints)
distributionoflabels.append(count4/numoftrainpoints)
distributionoflabels.append(count5/numoftrainpoints)

print distributionoflabels

#compute p(v|c)

p_v_given_c=[[] for i in range(6)]
for i in range(6):
	for v in range(k):
		p=float(map_v_given_c[i].count(v))/float(len(map_v_given_c[i]))
		if p==0:
			p_v_given_c[i].append(0.000000000001)
		else:
			p_v_given_c[i].append(p)

print p_v_given_c


##############################################

#global parameters
predicted_lables=[]
probabilities_eachpoint=[]




for i in range(numoftestpoints):
	global probabilities_eachpoint
	key=-1
	shortest=9999999999.0
	for v in range(0,k):
		distance_=distance(V[v],testpoints[i])      #############here points is in training set
       		if distance_<shortest:
        		shortest=distance_
        		key=v

	for c in range(6):
		probabilities_eachpoint.append(p_v_given_c[c][key]*distributionoflabels[c])
	predicted_lables.append(probabilities_eachpoint.index(max(probabilities_eachpoint)))
	probabilities_eachpoint=[]

print predicted_lables



# error rate computing 
def list_compare(lista,listb):
	minl=min(len(lista),len(listb))
	error_count=0
	for i in range(minl):
		if lista[i]==listb[i]:
			error_count=error_count
		else:
			error_count=error_count+1
	return float(error_count)/float(minl)




print list_compare(predicted_lables,testlabels)









#test
#f=open('Program_test.txt','w')
#f.write(trainlabels)
#for i in range(k):
#		for j in range(2):
#			f.write('%d' % v[i,j])
#			f.write("\t")
#		f.write(",")
#f.write('%d' % testlabels[i])
#f.close()
