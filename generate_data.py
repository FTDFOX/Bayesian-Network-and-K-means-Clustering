#!/usr/bin/python

#
# CSE 5522: HW 2 Data Generator
# Author: Eric Fosler-Lussier
# 
# This script can be used to generate data, sampling from n gaussian distributions
#
# Global parameters are set at the top of the script.

from numpy.random import *
from numpy import *
from matplotlib.pyplot import *
import json

# Global parameters
dims=2
nmeans=6
gaussianwidth=3
scale=10
deviationpct=.1
ntrainpoints=500
ntestpoints=120
colors=['b+','r+','g+','m+','k+','b*'];

#
# Functions
#

# generate 'num' gaussan means of 'dim' dimensionality 
def gen_means(num,dim):
    return randn(num,dim)*scale

# generate prior distribution over classes
# we do this by computing a random deviation from 0.5 for each class, and then
# normalize to be a proper distribution

def gen_distribution(num):
    values=(rand(num)*deviationpct-0.05)+0.5
    dist=values.copy()/sum(values)
    return dist

#
# Main part of script
#

# get means for data classes
means=gen_means(nmeans,dims)
# get priors over classes
dist=gen_distribution(nmeans)
# training,test labels are sampled from the multinomial distribution described
# by the class priors
trainlabels=argmax(multinomial(1,dist,ntrainpoints),axis=1)
testlabels=argmax(multinomial(1,dist,ntestpoints),axis=1)

# generate training points by sampling gaussians
trainpoints=randn(ntrainpoints,dims)*gaussianwidth
# i'm sure there was a clever way to do this in numpy but I didn't figure
# it out...  (email me if you know a non-looping way, I'm curious)
for i in xrange(0,ntrainpoints):
    trainpoints[i]+=means[trainlabels[i]]

# generate testing points
testpoints=randn(ntestpoints,dims)*gaussianwidth
for i in xrange(0,ntestpoints):
    testpoints[i]+=means[testlabels[i]]

# this is a quick way to look at the data.  Note that colors needs to be 
# at least as long as the number of classes        
for i in xrange(0,nmeans):
    tp=compress(trainlabels==i,trainpoints,axis=0)
    plot(tp[:,0],tp[:,1],colors[i])
show(block=False)
    
# dump data in both json and text formats
#
f=open('training.json','w')
json.dump({'points': trainpoints.tolist(), 'labels': trainlabels.tolist()},f)
f.close()

f=open('training.txt','w')
for i in xrange(0,ntrainpoints):
    f.write('%d' % trainlabels[i])
    f.write("\t")
    f.write("\t".join(['%.6f' % num for num in trainpoints[i]]))
    f.write("\n")
f.close()

f=open('testing.json','w')
json.dump({'points': testpoints.tolist(), 'labels': testlabels.tolist()},f)
f.close()

f=open('testing.txt','w')
for i in xrange(0,ntestpoints):
    f.write('%d' % testlabels[i])
    f.write("\t")
    f.write("\t".join(['%.6f' % num for num in testpoints[i]]))
    f.write("\n")
f.close()

            
