#!/bin/bash

#echo "In benchs"
#echo $PWD

line=$(($AWS_BATCH_JOB_ARRAY_INDEX + 1))
line=1

benches=()
#benches+=$(sed "$line!d" benchmarks.txt)
benches1=$(sed -n '70,+0p' benchmarks.txt)
IFS=$'\n' benches=( $benches1 )

for vars in "${benches[@]}"
do
	origin="${PWD}/${vars}"
	echo $origin

	x=$(find|grep "# gNh55" "${origin}") 	
	sed -i "s/$x/  4 0e00  # gNh55/" "${origin}"
	
	#x=$(find|grep "# gNh56" "${origin}")
        #sed -i "s/$x/  5 2.000000e-1  # gNh56/" "${origin}"

	ids=$(sed 's|.dat|''|g' <<< "$vars")
	ids=$(sed 's|.*/param_card.SeesawSM|''|g' <<< "$ids")
	#echo $ids

	IFS="." read -r -a array <<< "$ids"
	
	mass=${array[0]}
	alpha=${array[1]}

	#echo "$mass $alpha"

	bash param_distZH.sh "" "M${mass}" "Alpha${alpha}" "${origin}" "$1" "$2"

done

