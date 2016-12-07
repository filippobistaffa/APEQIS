#!/usr/bin/env bash

t=scalefree	# network topology

cd ..

for ns in `cut exp/runtime/LSE-CPU-SEQ.csv -f 1,2 -d ,`
do
	n=${ns%,*}
	s=${ns#*,}
	out=`./apeqis.sh -t $t -n $n -s $s 2> /dev/null`
	echo $n,$s,$out
done

cd exp
