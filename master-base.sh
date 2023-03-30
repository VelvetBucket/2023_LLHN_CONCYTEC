#Script para asignar un valor de coupling the Higgs a n5 n5 en todos los param cards.
#!/bin/bash

/home/cristian/Programs/MG5_aMC_v2_9_2/bin/mg5_aMC mg5_launches.txt

bash benchs-base.sh 
#bash hepmc_dist-base.sh 
