#Script para asignar un valor de coupling the Higgs a n5 n5 en todos los param cards.
#!/bin/bash

#echo "Analysis master"

echo "01"
python 01_cases_hepmc_reader.py
echo "02"
python 02_cases_photons_data.py
echo "03"
python 03_making_complete_hepmc.py
echo "04"
python 04_run_Delphes.py "$3" "$2"
echo "05"
python 05_extracting_root.py 
echo "06"
python 06_bins_after_Delphes.py "$1"
