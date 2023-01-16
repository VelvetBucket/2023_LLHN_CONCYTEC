#!/bin/bash

folder="${PWD}/param_c"
input="./widths.txt"

while IFS= read -r line
do
	width=( $line )
 	#echo "${width[1]}"
  	params=( $(ls "${folder}") )
  	for card in "${params[@]}"
  	do
		#echo "${card}"
		sed -i "193s/.*/DECAY  9900014   ${width[1]}/" "${folder}/${card}"
		mass="${card#*param_card_}"    ## trim from the front (left)
		mass="${mass%.dat*}"    ## trim from the back (right)

		bash param_dist.sh "${width[0]}" "${mass}" "${folder}/${card}"
	done
  #sed -i "193s/.*/${stringarray[1]}/" "${filename}"
done < "$input"
