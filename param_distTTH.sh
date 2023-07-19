#!/bin/bash

function changing () {
	x=$(find|grep "$1" "${run_path}")
	sed -i "s/$x/$2/g" "${run_path}" > /dev/null 2>&1
	#echo "$x"
}

function run_mg5 () {
	for tev in $tevs
		do
		tev_="$((tev*1000/2))"
		# Define las energias de los beams en el run_card
		beam1="     ${tev_}.0     = ebeam1  ! beam 1 total energy in GeV"
		beam2="     ${tev_}.0     = ebeam2  ! beam 2 total energy in GeV"
		changing " = ebeam1 " "$beam1"
		changing " = ebeam2 " "$beam2"
		
		# Le da el tag apropiado al run
		tag="  ${channel}_${windex}_${mindex}_${tindex}_${tev}     = run_tag ! name of the run "
		#echo $tag
		#exit 0
		changing " = run_tag " "$tag"
			
		#Copia el param_card correspondiente
		filename_d="${folder_destiny}/param_card.dat"
		cp "${filename_o}" "${filename_d}" 
			
		# Correr el run
		cd "${folder_destiny}"
		cd ..
		./bin/madevent "${config_path}" > /dev/null 2>&1
	done
}

echo "param_dist"

windex="$1"
mindex="$2"
tindex="$3"
filename_o="$4"
config_path="${PWD}/HN_run_config.txt"

tevs="13"

small="  1e-12 = small_width_treatment"
nevents="  ${5} = nevents ! Number of unweighted events requested "
ct="  0 = time_of_flight ! threshold (in mm) below which the invariant livetime is not written (-1 means not written)"
decay="   True  = cut_decays    ! Cut decay products "

pta_min=" 10.0  = pta       ! minimum pt for the photons "
ptl_min=" 10.0  = ptl       ! minimum pt for the charged leptons "
ptl_min_WH=" 5.0  = ptl       ! minimum pt for the charged leptons "
ptj_min=" 25.0  = ptj       ! minimum pt for the jets "
etaa_max=" 2.4  = etaa    ! max rap for the photons "
etal_max="# 2.5  = etal    ! max rap for the charged leptons"
etapdg_max=" {11: 2.5, 13: 2.7, 15: 5.0} = eta_max_pdg ! rap cut for other particles (syntax e.g. {6: 2.5, 23: 5})"
ptcl_min=" 27.0  = xptl ! minimum pt for at least one charged lepton "
etaj_max=" -1.0 = etaj    ! max rap for the jets "
drjj_min=" 0.0 = drjj    ! min distance between jets "
drjl_min=" 0.0 = drjl    ! min distance between jet and lepton "
r0gamma="  0.0 = R0gamma ! Radius of isolation code"
 
###################

tipos="TTH"

for channel in ${tipos}
	do
	folder_destiny="${6}/val-HN_${channel}/Cards"
	run_path="${folder_destiny}/run_card.dat"
	
	changing " = small_width_treatment "  "$small"
	changing " = nevents "  "$nevents"
	changing " = time_of_flight "  "$ct"
	changing " = cut_decays "  "$decay"
	changing " = pta "  "$pta_min"
	changing " = ptl "  "$ptl_min"
	if [ $channel == "WH" ]
		then
		changing " = ptl "  "$ptl_min_WH"
	fi
	changing " = ptj "  "$ptj_min"
	changing " = etaa "  "$etaa_max"
	changing " = etal "  "$etal_max"
	changing " = eta_max_pdg "  "$etapdg_max"
	changing " = xptl"  "$ptcl_max"
	changing " = etaj " "$etaj_max"
	changing " = drjj " "$drjj_min"
	changing " = drjl " "$drjl_min"
	changing " = R0gamma " "$r0gamma"
	
	run_mg5 "$channel"

done
