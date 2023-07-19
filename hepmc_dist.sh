#!/bin/bash

echo "hepmc dist"
#echo $PWD

folder_destiny="${2}/scripts_2208/data/raw"

tipos="ZH WH TTH"

mkdir -p "${folder_destiny}"

for tipo in ${tipos}
	do
	#declare -a arr
	folder_origin="${1}/val-HN_${tipo}/Events"
	cd ${folder_origin} > /dev/null 2>&1
	runs=( $(ls -d */) )
	for run in "${runs[@]}"
		do
		#echo "${run}"
		cd "${run}"
		count="$(ls -1 *.hepmc 2>/dev/null | wc -l)"
		#echo "${count}"
		if [ $count == 0 ]
			then
			#echo "hola"
			file_gz=("$(ls -d *.hepmc.gz)")
			gzip -dk "${file_gz}"
		fi
		file_mc=("$(ls -d *.hepmc)")
		#echo "${file_mc}"
		file_final="$(echo "${file_mc}" | sed 's/_pythia8_events//')"
		mv "${file_mc}" "${folder_destiny}/run_${file_final}"	
		cd ..
	done
done
