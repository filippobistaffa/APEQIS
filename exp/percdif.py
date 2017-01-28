import numpy
import math
import sys

n_n=5
n_rep=100

if ((len(sys.argv) != 6) and (len(sys.argv) != 8)):
	sys.exit("Usage: percdif.py input_csv1 input_csv2 col1 col2 output_csv [ n_n n_rep ]")

col1 = int(sys.argv[3])
col2 = int(sys.argv[4])

if (len(sys.argv) == 8):
	n_n = int(sys.argv[6])
	n_rep = int(sys.argv[7])
	n_rows = n_n * n_rep
	n=numpy.loadtxt(open(sys.argv[1],"rb"),delimiter=",")[0:n_rows,0]
	x=numpy.loadtxt(open(sys.argv[1],"rb"),delimiter=",")[0:n_rows,col1]
	y=numpy.loadtxt(open(sys.argv[2],"rb"),delimiter=",")[0:n_rows,col2]
else:
	n=numpy.loadtxt(open(sys.argv[1],"rb"),delimiter=",")[:,0]
	x=numpy.loadtxt(open(sys.argv[1],"rb"),delimiter=",")[:,col1]
	y=numpy.loadtxt(open(sys.argv[2],"rb"),delimiter=",")[:,col2]

n = n.reshape([n_n,n_rep])
n = n.mean(1)

pd = 100*(x-y)/x
pd = pd.reshape([n_n,n_rep])

mean = pd.mean(1)
std = pd.std(1)/math.sqrt(n_rep)
csvtxt=numpy.transpose(numpy.vstack((n,(mean,std))))

if sys.argv[5] == "-":
	print csvtxt
else:
	numpy.savetxt(sys.argv[5], csvtxt, delimiter=",", fmt="%f")
