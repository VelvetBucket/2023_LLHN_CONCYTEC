#!/bin/bash

origin="${PWD}/param_c/param_card_VALIDATION.dat"
destiny="${PWD}/param_c/param_card_VALIDATION1.dat"
benches=("60 0.5 2" "50 30 2" "40 20 2" "30 0.5 2")
const=6.58212196173*10^-16

for vars in "${benches[@]}"
do
	IFS=" " read -r -a array <<< "$vars"
	
	nlsp0=${array[0]}
	nlsp=$(awk "BEGIN {printf \"%.8E\n\", $nlsp0}" )
	
	lsp0=${array[1]}
	lsp1="${lsp0/./,}"
	lsp=$(awk "BEGIN {printf \"%.8E\n\", $lsp0}" )
	
	t0=${array[2]}
	width=$(awk "BEGIN {printf \"%.8E\n\", $const/$t0}" )
	
	sed -e "s/M_NLSP/${nlsp}/g" -e "s/M_LSP/${lsp}/g" -e "s/LFT_WIDTH/${width}/g" "${origin}" > "${destiny}"
	
	bash param_dist.sh "MN${nlsp0}" "ML${lsp1}" "T${t0}" "${destiny}"
	
done
