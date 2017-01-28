#!/usr/bin/env bash

d=10		# driver's percentage
is=0		# initial seed
t=scalefree	# network topology
nt=40		# number of tests
n=20		# number of agents

# OUTPUT = W,SEED,ORIG_SOL_VALUE,SR-CFSS_RUNTIME,SR-CFSS_VIS_NODES,CFSS_SOL_VALUE,CFSS_RUNTIME,CFSS_VIS_NODES,ISG_SOL_VALUE

isg=`mktemp`
isgsol=`mktemp`

cd ..

for w in 10
do
	i=0
	s=$is

	while [ $i -lt $nt ]
	do
		ape=`./apeqis.sh -t $t -n $n -d $d -s $s -c $isg -w $w 2> /dev/null`

		if [[ $? == 0 ]]
		then
			cd SR-CFSS
			srcfss=`./sr.sh -t $t -n $n -d $d -s $s 2> /dev/null`
			cd ..

			#echo SR-CFSS: $srcfss

			cd CFSS
			cfss=`./cfss.sh -t $isg -o $isgsol 2> /dev/null`
			cd ..

			#echo CFSS: $cfss

			cd SR-VALUE
			srvalue=`./srvalue.sh -i $isgsol -s $s -c 2> /dev/null`
			cd ..

			#echo SR-VALUE: $srvalue

			echo $w,$s,$srcfss,$cfss,$srvalue
			i=$(( $i + 1 ))
		fi

		s=$(( $s + 1 ))
	done
done

rm $isg
rm $isgsol
cd exp
