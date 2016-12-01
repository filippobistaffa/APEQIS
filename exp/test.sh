#!/usr/bin/env bash

is=0		# initial seed
t=scalefree	# network topology
nt=20		# number of tests

cd ..

for n in 100 200 300
do
	i=0
	s=$is

	while [ $i -lt $nt ]
	do
		out=`./apeqis.sh -t $t -n $n -s $s 2> /dev/null`

		if [[ $? == 0 ]]
		then
			echo $n,$s,$out
			i=$(( $i + 1 ))
		fi

		s=$(( $s + 1 ))
	done
done

cd exp
