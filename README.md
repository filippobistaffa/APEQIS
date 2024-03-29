<p align="center"><img src="https://filippobistaffa.github.io/images/apeqis.svg" width="500" /></p>

APEQIS: APproximately EQuivalent IS-represented cooperative games
===================
This repository contains the implementation of the algorithm presented by Filippo Bistaffa, Georgios Chalkiadakis, and Alessandro Farinelli in “[Efficient Coalition Structure Generation via Approximately Equivalent Induced Subgraph Games](	http://hdl.handle.net/10261/223529)”, IEEE Transactions on Cybernetics (IEEE TCYB) volume 52 issue 6, 2021, DOI: [10.1109/TCYB.2020.3040622](https://doi.org/10.1109/TCYB.2020.3040622).


Requirements
----------
APEQIS requires `g++` to compile, and relies on [IBM CPLEX](https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer) to solve the LP model. In order to compile against CPLEX, `CPLEXROOT` inside [`Makefile`](Makefile) must be set to point to the root folder of CPLEX.

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
