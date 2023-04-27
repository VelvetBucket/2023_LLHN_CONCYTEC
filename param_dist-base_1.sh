#Script para asignar un valor de coupling the Higgs a n5 n5 en todos los param cards.
#!/bin/bash
function changing () {
	x=$(find|grep "$1" "${run_path}")
	sed -i "s/$x/$2/g" "${run_path}"
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
		changing " = run_tag " "$tag"
			
		#Copia el param_card correspondiente
		filename_d="${folder_destiny}/param_card.dat"
		cp "${filename_o}" "${filename_d}" 


		#Cambia el small_width_treatment
		#sed -i "17s/.*/${small}/" "${run_path}"

		#Seteando numero de eventos
		#sed -i "26s/.*/${nevents}/" "${run_path}"

		#Agrega los cortes comunes
		#sed -i "59s/.*/${ct}/" "${run_path}"
		#sed -i "94s/.*/${decay}/" "${run_path}"
			
		# Correr el run
		cd "${folder_destiny}"
		cd ..
		./bin/madevent "${config_path}"
	done
}

windex="$1"
mindex="$2"
tindex="$3"
filename_o="$4"
config_path='/home/cristian/Programs/MG5_aMC_v2_9_2/HN_run_config-base.txt'

tevs="13"

small="  1e-12 = small_width_treatment"
nevents="  1000 = nevents ! Number of unweighted events requested "
ct="  0 = time_of_flight ! threshold (in mm) below which the invariant livetime is not written (-1 means not written)"
decay="   False  = cut_decays    ! Cut decay products "

pta_min=" 0.0  = pta       ! minimum pt for the photons "
ptl_min=" 0.0  = ptl       ! minimum pt for the charged leptons "
ptl_min_WH=" 0.0  = ptl       ! minimum pt for the charged leptons "
ptj_min=" 0.0  = ptj       ! minimum pt for the jets "
etaa_max=" -1.0  = etaa    ! max rap for the photons "
etal_max=" -1.0 = etal    ! max rap for the charged leptons"
ptcl_min=" 0.0  = xptl ! minimum pt for at least one charged lepton "
etaj_max=" -1.0 = etaj    ! max rap for the jets "
drjj_min=" 0.0 = drjj    ! min distance between jets "
drjl_min=" 0.0 = drjl    ! min distance between jet and lepton "
r0gamma="  0.0 = R0gamma ! Radius of isolation code"


############## WH ##################

tipos="TTH"

for channel in ${tipos}
	do
	folder_destiny="/home/cristian/Programs/MG5_aMC_v2_9_2/val-cs-1-HN_${channel}/Cards"
	run_path="${folder_destiny}/run_card.dat"
	
	changing " = small_width_treatment "  "$small"
	changing " = nevents "  "$nevents"
	changing " = time_of_flight "  "$ct"
	changing " = cut_decays "  "$decay"
	changing " = pta "  "$pta_min"
	changing " = ptl "  "$ptl_min"
	changing " = ptj "  "$ptj_min"
	changing " = etaa "  "$etaa_max"
	changing " = etal "  "$etal_max"
	changing " = xptl "  "$ptcl_max"
	changing " = etaj " "$etaj_max"
	changing " = drjj " "$drjj_min"
	changing " = drjl " "$drjl_min"
	changing " = R0gamma " "$r0gamma"
	
	run_mg5 "$channel"
done



