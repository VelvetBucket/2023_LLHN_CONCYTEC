#Script para asignar un valor de coupling the Higgs a n5 n5 en todos los param cards.
#!/bin/bash

export AWS_BATCH_JOB_ARRAY_INDEX=4

echo $AWS_BATCH_JOB_ARRAY_INDEX
 
madgraph_folder="/Collider/MG5_aMC_v2_9_11"
delphes_folder="/Collider/MG5_aMC_v2_9_11/Delphes"
destiny_folder="/Collider"

sed "s|FOLDER|$madgraph_folder|g" mg5_launches.txt > mg5_launches_proper.txt

${madgraph_folder}/bin/mg5_aMC mg5_launches_proper.txt

bash benchs.sh "$1" "$madgraph_folder"
bash hepmc_dist.sh "$madgraph_folder" "$destiny_folder"
bash crossec_dist.sh "$destiny_folder" "$madgraph_folder"

source ~/.bashrc

cd ./scripts_2208/
#echo $PYTHONPATH
bash analysis_master.sh "$1" "$delphes_folder"  "$destiny_folder"
echo "Done!"
