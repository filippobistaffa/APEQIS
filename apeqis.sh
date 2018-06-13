#!/usr/bin/env bash

red='\033[0;31m'				# Red
nc='\033[0m'					# No color
re='^[0-9]+$'					# Regular expression to detect natural numbers
fn_re='^.*Agents[0-9]+Rules[0-9]+.*.txt$'	# Regular expression to match Ueda's filename format
c=""
r=""

usage() { echo -e "Usage: $0 -i <mcnet_file> [-w <weight>] [-c <out_file>] [-r <res_file>] [-f]\n-i\tMC-net input file (filename must be formatted as Agents<n_agents>Rules<n_Rules>*.txt)\n-w\tWeight for singletons in weighted norm (optional, default = 1)\n-c\tOutputs an input file formatted for CFSS (optional)\n-r\tWrites the residual vector to file (optional)\n-f\tUse a fully connected graph (optional)" 1>&2; exit 1; }

while getopts ":i:c:w:r:f" o; do
	case "${o}" in
	i)
		i=${OPTARG}
		if [ ! -f "$i" ]
		then
			echo -e "${red}Input file \"$i\" not found!${nc}\n" 1>&2
			usage
		fi
		if ! [[ $i =~ $fn_re ]] ; then
			echo -e "${red}Input filename \"$i\" not in the correct format!${nc}\n" 1>&2
			usage
		fi
		;;
	w)
		w=${OPTARG}
		if ! [[ $w =~ $re ]]
		then
			echo -e "${red}Parameter w must be a number!${nc}\n" 1>&2
			usage
		fi
		;;
	c)
		c=${OPTARG}
		touch $c 2> /dev/null
		rc=$?
		if [[ $rc != 0 ]]
		then
			echo -e "${red}Unable to create $c${nc}" 1>&2
			exit
		else
			rm $c
		fi
		;;
	r)
		r=${OPTARG}
		touch $r 2> /dev/null
		rc=$?
		if [[ $rc != 0 ]]
		then
			echo -e "${red}Unable to create $r${nc}" 1>&2
			exit
		else
			rm $r
		fi
		;;
	f)
		f=1
		;;
	\?)
		echo -e "${red}-$OPTARG is not a valid option!${nc}\n" 1>&2
		usage
		;;
	esac
done
shift $((OPTIND-1))

if [ -z "${i}" ]
then
	echo -e "${red}Missing input file!${nc}\n" 1>&2
	usage
fi

n=${i##*Agents}
n=${n%Rules*}
rul=${i##*Rules}
end=`echo $rul | grep -o [[:alpha:]] | head -n 1`
rul=${rul%${end}*}

tmp=`mktemp`
echo "#define _N $n" > $tmp
echo "#define _D $n" >> $tmp
echo "#define CORES `grep -c ^processor /proc/cpuinfo`" >> $tmp
echo "#define RULES $rul" >> $tmp

if [ ! -z "${w}" ]
then
	echo "#define WEIGHT $w" >> $tmp
fi

if [ ! -z "${c}" ]
then
	echo "#define CFSS" >> $tmp
else
	c="x"
fi

if [ ! -z "${r}" ]
then
	echo "#define RESIDUAL" >> $tmp
else
	r="x"
fi

if [ ! -z "${f}" ]
then
	echo "#define FULL_GRAPH" >> $tmp
fi

if [ ! -f instance.h ]
then
	mv $tmp "instance.h"
else
	md5a=`md5sum instance.h | cut -d\  -f 1`
	md5b=`md5sum $tmp | cut -d\  -f 1`

	if [ $md5a != $md5b ]
	then
		mv $tmp "instance.h"
	else
		rm $tmp
	fi
fi

make -j
if [[ $? == 0 ]]
then
	./apeqis $i $c $r
	rc=$?
fi

exit $rc
