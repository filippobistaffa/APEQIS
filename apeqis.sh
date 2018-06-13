#!/usr/bin/env bash

red='\033[0;31m'				# Red
nc='\033[0m'					# No color
re='^[0-9]+$'					# Regular expression to detect natural numbers
fn_re='^.*Agents[0-9]+Coalitions[0-9]+.*.txt$'	# Regular expression to match Ueda's filename format

usage() { echo -e "Usage: $0 -i <mcnet_file> [-c <out_file>] [-w <weight>] [-f]\n-i\tMC-net input file (filename must be formatted as Agents<n_agents>Coalitions<n_coalitions>*.txt)\n-c\tOutputs an input file formatted for CFSS (optional)\n-w\tWeight for singletons in weighted norm (optional, default = 1)\n-f\tUse a fully connected graph (optional)" 1>&2; exit 1; }

while getopts ":i:c:w:f" o; do
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
		if ! [[ $w =~ $re ]] ; then
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
n=${n%Coalitions*}
r=${i##*Coalitions}
end=`echo $r | grep -o [[:alpha:]] | head -n 1`
r=${r%${end}*}

tmp=`mktemp`
echo "#define N $n" > $tmp
echo "#define RULES $r" >> $tmp

if [ ! -z "${c}" ]
then
	echo "#define CFSS \"$c\"" >> $tmp
fi

if [ ! -z "${w}" ]
then
	echo "#define WEIGHT $w" >> $tmp
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
	./apeqis $i
	rc=$?
fi

exit $rc
