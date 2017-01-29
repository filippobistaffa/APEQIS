import numpy
import math
import sys

n_n=5
n_rep=100

if ((len(sys.argv) != 4) and (len(sys.argv) != 6)):
	sys.exit("Usage: speedup.py input_csv1 input_csv2 output_csv [ n_n n_rep ]")

if (len(sys.argv) == 6):
	n_n = int(sys.argv[4])
	n_rep = int(sys.argv[5])
	n_rows = n_n * n_rep
	x=numpy.loadtxt(open(sys.argv[1],"rb"),delimiter=",")[0:n_rows,:]
	y=numpy.loadtxt(open(sys.argv[2],"rb"),delimiter=",")[0:n_rows,-1]
else:
	x=numpy.loadtxt(open(sys.argv[1],"rb"),delimiter=",")
	y=numpy.loadtxt(open(sys.argv[2],"rb"),delimiter=",")[:,-1]

n = x[:,0]
n = n.reshape([n_n,n_rep])
n = n.mean(1)
x = x[:,-1]

sp=x/y
sp=sp.reshape([n_n,n_rep])

mean = sp.mean(1)
std = sp.std(1)/math.sqrt(n_rep)
csvtxt=numpy.transpose(numpy.vstack((n,(mean,std))))

if sys.argv[3] == "-":
	print csvtxt
else:
	numpy.savetxt(sys.argv[3], csvtxt, delimiter=",", fmt="%f")
