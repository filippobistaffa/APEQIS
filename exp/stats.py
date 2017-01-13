import numpy
import math
import sys

n_n=5
n_rep=100

if ((len(sys.argv) != 4) and (len(sys.argv) != 6)):
	sys.exit("Usage: stats.py input_csv column output_csv [ n_n n_rep ]")

if (len(sys.argv) == 6):
	n_n = int(sys.argv[4])
	n_rep = int(sys.argv[5])
	n_rows = n_n * n_rep
	x=numpy.loadtxt(open(sys.argv[1],"rb"),delimiter=",")[0:n_rows,:]
else:
	x=numpy.loadtxt(open(sys.argv[1],"rb"),delimiter=",")

#print x

n = x[:,0]
n = n.reshape([n_n,n_rep])
u = x[:,int(sys.argv[2])]
v = u.reshape([n_n,n_rep])

n = n.mean(1)
mean = v.mean(1)
std = v.std(1)/math.sqrt(n_rep)

csvtxt=numpy.transpose(numpy.vstack((n,(mean,std))))

if sys.argv[3] == "-":
	print csvtxt
else:
	numpy.savetxt(sys.argv[3], csvtxt, delimiter=",", fmt="%f")
