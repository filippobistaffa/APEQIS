APEQIS: APproximately EQuivalent IS-represented cooperative games
===================

<p align="center"><img src="https://filippobistaffa.github.io/images/apeqis.svg" width="500" /></p>

Requirements
----------
APEQIS requires `g++` and [CUDA](http://www.nvidia.com/object/cuda_home_new.html) to compile and execute. CUDA 6.0 or newer is required as APEQIS uses [Unified Memory](https://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/). Before compiling and executing APEQIS, it is necessary to adjust the `CUDAARCH` compilation parameter (i.e., the compute capabilities of the GPU) within the [`Makefile`](Makefile). The correct value can be printed by means of the `smquery` utility.

APEQIS has been tested on an [NVIDIA GeForce GTX TITAN X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x) GPU.

In order to employ Twitter as network topology, `java` must be installed on the system, and the [Twitter GitHub repository](https://github.com/filippobistaffa/twitter) must be `git clone`'d inside APEQIS's root directory.

Execution
----------
APEQIS must be executed by means of the [`apeqis.sh`](apeqis.sh) script, i.e.,
```
./apeqis.sh -t <scalefree|twitter> -n <#agents> -s <seed> [-m <barab_m>] [-d <drivers%>] [-c <out_file>]

-t	Network topology (either scalefree or twitter)
-n	Number of agents
-s	Seed
-d	Drivers' percentage (optional, default d = 20)
-m	Parameter m of the Barabasi-Albert model (optional, default m = 2)
-c	Outputs an input file formatted for CFSS (optional)
```

Acknowledgements
----------
APEQIS employs the [GeoLife dataset by Microsoft Research](http://research.microsoft.com/en-us/projects/geolife) presented by Yu Zheng, Quannan Li, Yukun Chen, Xing Xie, and Wei-Ying Ma in “[Understanding mobility based on GPS data](https://www.microsoft.com/en-us/research/publication/understanding-mobility-based-on-gps-data)”, Proceedings of the 10th ACM conference on Ubiquitous Computing (Ubicomp), pages 312−321, 2008, ACM press.
