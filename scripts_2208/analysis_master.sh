#Script para asignar un valor de coupling the Higgs a n5 n5 en todos los param cards.
#!/bin/bash

python 01_cases_hepmc_reader.py
python 02_cases_photons_data.py
python 03_making_complete_hepmc.py
python 04_run_Delphes.py "$3" "$2"
python 05_extracting_root.py 
python 06_bins_after_Delphes.py "$1"
