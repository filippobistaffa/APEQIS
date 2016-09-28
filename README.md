LP-ISG: a Linear Programming model for Induced Subgraph Games
===================

Requirements
----------
LP-ISG requires `g++` to compile, and relies on [IBM CPLEX](https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer) to solve the LP model. In order to compile against CPLEX, `INC` and `LDIR` inside [`Makefile`](https://github.com/filippobistaffa/LP-ISG/blob/master/Makefile)  must be set to point to the `include` and `lib` folders of CPLEX.

In order to employ Twitter as network topology, `java` must be installed on the system, and the [Twitter GitHub repository](https://github.com/filippobistaffa/twitter) must be `git clone`'d inside LP-ISG's root directory.

Execution
----------
LP-ISG must be executed by means of the [`lpisg.sh`](https://github.com/filippobistaffa/LP-ISG/blob/master/lpisg.sh) script, i.e.,
```
./lpisg.sh -t <scalefree|twitter> -n <#agents> -s <seed> [-m <barab_m>] [-d <drivers_%>] [-c <out_file>]

-t	Network topology (either scalefree or twitter)
-n	Number of agents
-s	Seed
-d	Drivers' percentage (optional, default d = 20)
-m	Parameter m of the Barabasi-Albert model (optional, default m = 2)
-c	Outputs a solution file formatted for CFSS (optional)
```

Acknowledgements
----------
LP-ISG employs the [GeoLife dataset by Microsoft Research](http://research.microsoft.com/en-us/projects/geolife) presented by Yu Zheng, Quannan Li, Yukun Chen, Xing Xie, and Wei-Ying Ma in “[Understanding mobility based on GPS data](https://www.microsoft.com/en-us/research/publication/understanding-mobility-based-on-gps-data)”, Proceedings of the 10th ACM conference on Ubiquitous Computing (Ubicomp), pages 312−321, 2008, ACM press.
