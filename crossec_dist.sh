#Script para mandar los hepmc de los runs a la carpeta correspondiente
#!/bin/bash

folder_destiny="${1}/scripts_2208/data/raw"

tipos="ZH WH TTH"

echo '' > ./scripts_2208/data/cross_section.dat

for tipo in ${tipos}
	do
	#declare -a arr
	folder_origin="${2}/val-HN_${tipo}/Events"
	cd ${folder_origin}
	runs=( $(ls -d */) )
	for run in "${runs[@]}"
		do
		cd "${run}"
		file_mc=("$(ls -d *_banner.txt)")
		run="${run::-1}_"
		cross=$(find|grep "Integrated" "${file_mc}")
		cross=$(sed 's| Integrated weight (pb) |''|g' <<<"$cross")
		cross=$(sed 's|\#|''|g' <<<"$cross")
		cross=$(sed 's|\: |''|g' <<<"$cross")
		file_mc="${file_mc/_banner.txt/''}"
		file_mc="${file_mc/$run/''}"
		echo "${file_mc}	${cross}" >> "${1}/scripts_2208/data/cross_section.dat"
		cd ..
	done
done
