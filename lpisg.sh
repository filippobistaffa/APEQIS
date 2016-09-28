#!/usr/bin/env bash

red='\033[0;31m'			# Red
nc='\033[0m'				# No color
re='^[0-9]+$'				# Regular expression to detect natural numbers
d=20					# Default d = 20
m=2					# Default m = 2
basename="twitter/net/twitter-2010"	# Twitter files must be in "twitter/net" subdir and have twitter-2010.* filenames
wg="twitter/wg"				# Place WebGraph libraries in "twitter/wg" subdir
tw=""

usage() { echo -e "Usage: $0 -t <scalefree|twitter> -n <#agents> -s <seed> [-m <barab_m>] [-d <drivers_%>] [-c <out_file>]\n-t\tNetwork topology (either scalefree or twitter)\n-n\tNumber of agents\n-s\tSeed\n-d\tDrivers' percentage (optional, default d = 20)\n-m\tParameter m of the Barabasi-Albert model (optional, default m = 2)\n-c\tOutputs a solution file formatted for CFSS (optional)" 1>&2; exit 1; }

while getopts ":t:n:s:d:m:c:" o; do
	case "${o}" in
	t)
		t=${OPTARG}
		if ! [[ $t == "scalefree" || $t == "twitter" ]] ; then
			echo -e "${red}Network topology must be either scalefree or twitter!${nc}\n"
			usage
		fi
		;;
	n)
		n=${OPTARG}
		if ! [[ $n =~ $re ]] ; then
			echo -e "${red}Number of agents must be a number!${nc}\n"
			usage
		fi
		;;
	s)
		s=${OPTARG}
		if ! [[ $s =~ $re ]] ; then
			echo -e "${red}Seed must be a number!${nc}\n"
			usage
		fi
		;;
	d)
		d=${OPTARG}
		if ! [[ $d =~ $re ]] ; then
			echo -e "${red}Drivers' percentage must be a number!${nc}\n"
			usage
		fi
		;;
	m)
		m=${OPTARG}
		if ! [[ $m =~ $re ]] ; then
			echo -e "${red}Parameter m must be a number!${nc}\n"
			usage
		fi
		;;
	c)
		c=${OPTARG}
		touch $c 2> /dev/null
		rc=$?
		if [[ $rc != 0 ]]
		then
			echo -e "${red}Unable to create $c${nc}"
			exit
		fi
		;;
	\?)
		echo -e "${red}-$OPTARG is not a valid option!${nc}\n"
		usage
		;;
	esac
done
shift $((OPTIND-1))

if [ -z "${t}" ] || [ -z "${n}" ] || [ -z "${s}" ]; then
	echo -e "${red}Missing one or more required options!${nc}\n"
	usage
fi

tmp=`mktemp`
echo "#define N $n" > $tmp
echo "#define DRIVERSPERC $d" >> $tmp
if [ ! -z "${c}" ]
then
	echo "#define CFSS \"$c\"" >> $tmp
fi

if [[ $t == "scalefree" ]]
then
	echo "#define M $m" >> $tmp
else
	echo "#define TWITTER" >> $tmp
	tw=`mktemp`
	java -Xmx4000m -cp .:$wg/* ReduceGraph $basename $n $s | grep -v WARN > $tw
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
	./lpisg $s $tw
fi

if [[ $t == "twitter" ]]
then
	rm $tw
fi
