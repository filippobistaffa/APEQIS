<p align="center"><img src="https://filippobistaffa.github.io/images/apeqis.svg" width="500" /></p>

APEQIS: APproximately EQuivalent IS-represented cooperative games
===================

Requirements
----------
APEQIS requires `g++` to compile, and relies on [IBM CPLEX](https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer) to solve the LP model. In order to compile against CPLEX, `CPLEXROOT` inside [`Makefile`](Makefile) must be set to point to the root folder of CPLEX.


Execution
----------
APEQIS must be executed by means of the [`apeqis.sh`](apeqis.sh) script, i.e.,
```
./apeqis.sh -i <mcnet_file> [-c <out_file>] [-w <weight>]

Usage: 
-i	MC-net input file (filename must be formatted as Agents<n_agents>Coalitions<n_coalitions>*.txt)
-c	Outputs an input file formatted for CFSS (optional)
-w	Weight for singletons in weighted norm
```
