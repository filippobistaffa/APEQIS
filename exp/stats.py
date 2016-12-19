import numpy
import math
import sys

x=numpy.loadtxt(open(sys.argv[1],"rb"),delimiter=",")
#print x

n = x[:,0]
n = n.reshape([5,100])
u = x[:,-1]
v = u.reshape([5,100])

n = n.mean(1)
mean = v.mean(1)
std = v.std(1)/math.sqrt(100)

csvtxt=numpy.transpose(numpy.vstack((n,(mean,std))))
numpy.savetxt(sys.argv[2], csvtxt, delimiter=",", fmt="%f")
