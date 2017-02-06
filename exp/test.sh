#!/usr/bin/env bash

is=0		# initial seed
t=scalefree	# network topology
nt=100		# number of tests
w=100		# singletons weight

cd ..

for n in 100 200 300 400 500
do
	i=0
	s=$is

	while [ $i -lt $nt ]
	do
		grep "^$n,$s," exp/$1 > /dev/null 2>&1

		if [[ $? != 0 ]]
		then
			out=`./apeqis.sh -t $t -n $n -s $s -w $w 2> /dev/null`

			if [[ $? == 0 ]]
			then
				echo $n,$s,$out
				i=$(( $i + 1 ))
			fi
		fi

		s=$(( $s + 1 ))
	done
done

cd exp
