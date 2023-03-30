#Script para mandar los hepmc de los runs a la carpeta correspondiente
#!/bin/bash

folder_destiny="/home/cristian/Desktop/HEP_Jones/paper_2023/scripts_2208/data/raw"

tipos="ZH WH TTH"

for tipo in ${tipos}
	do
	#declare -a arr
	folder_origin="/home/cristian/Programs/MG5_aMC_v2_9_2/val-cs-1-HN_${tipo}/Events"
	cd ${folder_origin}
	runs=( $(ls -d */) )
	for run in "${runs[@]}"
		do
		#echo "${run}"
		cd "${run}"
		file_mc=("$(ls -d *_banner.txt)")
		cross=("$(tail -4 ${file_mc} | head -1)")
		file_mc=${file_mc%_banner.txt*}
		file_mc=${file_mc#*("${run}")}
		echo "${file_mc}	${cross}"
		#file_final="$(echo "${file_mc}" | sed 's/_pythia8_events//')"
		#cp "${file_mc}" "${folder_destiny}/run_${file_final}"	
		cd ..
	done
done
