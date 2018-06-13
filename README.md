<p align="center"><img src="https://filippobistaffa.github.io/images/apeqis.svg" width="500" /></p>

APEQIS: APproximately EQuivalent IS-represented cooperative games
===================

Requirements
----------
APEQIS requires:
- `g++` with C++11 support.
- [Armadillo](http://arma.sourceforge.net) library installed.
- [CUDA](http://www.nvidia.com/object/cuda_home_new.html) 6.0 or newer, since APEQIS uses [Unified Memory](https://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/). Before compiling and executing APEQIS, it is necessary to adjust the `CUDAARCH` compilation parameter (i.e., the compute capabilities of the GPU) within the [`Makefile`](Makefile). The correct value can be printed by means of the `smquery` utility. APEQIS has been tested on an [NVIDIA GeForce GTX TITAN X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x).

Execution
----------
APEQIS must be executed by means of the [`apeqis.sh`](apeqis.sh) script, i.e.,
```
./apeqis.sh -i <mcnet_file> [-w <weight>] [-c <out_file>] [-r <res_file>] [-f]

-i	MC-net input file (filename must be formatted as Agents<n_agents>Coalitions<n_coalitions>*.txt)
-w	Weight for singletons in weighted norm (optional, default = 1)
-c	Outputs an input file formatted for CFSS (optional)
-r	Writes the residual vector to file (optional)
-f	Use a fully connected graph (optional)
```
