#!/bin/bash

cd 2023_LLHN_CONCYTEC
$SHELL

x1=100000

madgraph_folder="/Collider/MG5_aMC_v2_9_11"
delphes_folder="/Collider/MG5_aMC_v2_9_11/Delphes"
destiny_folder="/Collider"

#sed -i 's+run_mode = 2+run_mode = 0+' ${madgraph_folder}/input/mg5_configuration.txt
sed -i 's+nb_core = 4+nb_core = 2+' ${madgraph_folder}/input/mg5_configuration.txt

sed "s|FOLDER|$madgraph_folder|g" mg5_launches.txt > mg5_launches_proper.txt

${madgraph_folder}/bin/mg5_aMC mg5_launches_proper.txt > /dev/null 2>&1

bash benchsTTH.sh "$x1" "$madgraph_folder"
bash hepmc_dist.sh "$madgraph_folder" "$destiny_folder"
bash crossec_dist.sh "$destiny_folder" "$madgraph_folder"


echo "Before source"
source ~/.bashrc
echo "After source"

cd ./scripts_2208/
#echo $PYTHONPATH
bash analysis_master.sh "$x1" "$delphes_folder"  "$destiny_folder"
#echo "Done!"
