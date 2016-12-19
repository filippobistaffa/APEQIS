import numpy
import math
import sys

x=numpy.loadtxt(open(sys.argv[1],"rb"),delimiter=",")
y=numpy.loadtxt(open(sys.argv[2],"rb"),delimiter=",")[:,-1]

n = x[:,0]
n = n.reshape([5,100])
n = n.mean(1)
x = x[:,-1]

sp=x/y
sp=sp.reshape([5,100])

mean = sp.mean(1)
std = sp.std(1)/math.sqrt(100)
csvtxt=numpy.transpose(numpy.vstack((n,(mean,std))))
numpy.savetxt("/home/filippo/phd/papers/ae-isg/csv/speedup/"+sys.argv[1], csvtxt, delimiter=",", fmt="%f")
