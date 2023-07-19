#!/bin/bash

echo "crosssec"
echo $PWD

#tipos="ZH WH TTH"
tipos="ZH"


#echo '' > ./scripts_2208/data/cross_section.dat

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
cp ${1}/scripts_2208/data/cross_section.dat /Collider/2023_LLHN_CONCYTEC/
