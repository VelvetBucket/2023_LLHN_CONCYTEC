#Script para asignar un valor de coupling the Higgs a n5 n5 en todos los param cards.
#!/bin/bash

function run_mg5 () {
	for tev in $tevs
		do
		tev_="$((tev*1000/2))"
		# Define las energias de los beams en el run_card
		beam1="     ${tev_}.0     = ebeam1  ! beam 1 total energy in GeV"
		beam2="     ${tev_}.0     = ebeam2  ! beam 2 total energy in GeV"
		sed -i "35s/.*/${beam1}/" "${run_path}"
		sed -i "36s/.*/${beam2}/" "${run_path}"
		
		# Le da el tag apropiado al run
		tag="  ${channel}_${windex}_${mindex}_${tindex}_${tev}     = run_tag ! name of the run "
		sed -i "21s/.*/${tag}/" "${run_path}"
			
		#Copia el param_card correspondiente
		filename_d="${folder_destiny}/param_card.dat"
		cp "${filename_o}" "${filename_d}" 


		#Cambia el small_width_treatment
		sed -i "17s/.*/${small}/" "${run_path}"

		#Seteando numero de eventos
		sed -i "26s/.*/${nevents}/" "${run_path}"

		#Agrega los cortes comunes
		sed -i "59s/.*/${ct}/" "${run_path}"
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
decay="   True  = cut_decays    ! Cut decay products "

pta_min=" 10.0  = pta       ! minimum pt for the photons "
ptl_min=" 10.0  = ptl       ! minimum pt for the charged leptons "
ptl_min_WH=" 27.0  = ptl       ! minimum pt for the charged leptons "
ptj_min=" 25.0  = ptj       ! minimum pt for the jets "
etaa_max=" 2.4  = etaa    ! max rap for the photons "
etal_max="# 2.5  = etal    ! max rap for the charged leptons"
etal_max_TTH=" 2.7  = etal    ! max rap for the charged leptons"
etapdg_max=" {11: 2.5, 13: 2.7, 15: 5.0, 17: 5.0} = eta_max_pdg ! rap cut for other particles (syntax e.g. {6: 2.5, 23: 5})"
ptcl_min=" 27.0  = xptl ! minimum pt for at least one charged lepton "


############## WH ##################

channel="ZH"
folder_destiny="/home/cristian/Programs/MG5_aMC_v2_9_2/val-cs-HN_${channel}/Cards"
run_path="${folder_destiny}/run_card.dat"

# Agregrando cortes propios
sed -i "104s/.*/${etapdg_max}/" "${run_path}"
sed -i "155s/.*/${ptcl_min}/" "${run_path}"

run_mg5 "${channel}"

############## WH ##################

channel="WH"
folder_destiny="/home/cristian/Programs/MG5_aMC_v2_9_2/val-cs-HN_${channel}/Cards"
run_path="${folder_destiny}/run_card.dat"

# Agregrando cortes propios
sed -i "104s/.*/${etapdg_max}/" "${run_path}"

run_mg5 "${channel}"

############## TTH ##################

channel="TTH"
folder_destiny="/home/cristian/Programs/MG5_aMC_v2_9_2/val-cs-HN_${channel}/Cards"
run_path="${folder_destiny}/run_card.dat"

# Agregrando cortes propios


run_mg5 "${channel}"

