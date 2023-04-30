#Script para asignar un valor de coupling the Higgs a n5 n5 en todos los param cards.
#!/bin/bash

madgraph_folder="/home/cristian/Programs/MG5_aMC_v2_9_2"
delphes_folder="/home/cristian/Programs/MG5_aMC_v2_9_2/Delphes"
destiny_folder="/home/cristian/Desktop/HEP_Jones/paper_2023"

sed "s|FOLDER|$madgraph_folder|g" mg5_launches.txt > mg5_launches_proper.txt

${madgraph_folder}/bin/mg5_aMC mg5_launches_proper.txt

bash benchs.sh "$1" "$madgraph_folder"
bash hepmc_dist.sh "$madgraph_folder"
bash crossec_dist.sh "$destiny_folder" "$madgraph_folder"

cd ./scripts_2208/
bash analysis_master.sh "$1" "$delphes_folder"  "$destiny_folder"
