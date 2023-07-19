#!/bin/bash

#echo "In benchs"
#echo $PWD

line=$(($AWS_BATCH_JOB_ARRAY_INDEX + 1))
#line=2
benches=()
benches+=$(sed "$line!d" benchmarks.txt)

for vars in "${benches[@]}"
do
	origin="${PWD}/${vars}"
	#echo $origin
	
	ids=$(sed 's|.dat|''|g' <<< "$vars")
	ids=$(sed 's|param_c/param_card.SeesawSM|''|g' <<< "$ids")
	echo $ids

	IFS="." read -r -a array <<< "$ids"
	
	mass=${array[0]}
	alpha=${array[1]}

	#echo "$mass $alpha"

	bash param_distZH.sh "" "M${mass}" "Alpha${alpha}" "${origin}" "$1" "$2"

done

