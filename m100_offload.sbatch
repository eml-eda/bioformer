#!/bin/bash

# Load module, venv and stuff
module load profile/deeplrn
module load autoload cineca-ai/2.2.0
source $WORK/dl/bin/activate

sbatch  --account IscrC_AdHeNAS --job-name bioformer-$1-$2-$3-$4-$5-$6-$7-${10}-${13}-${14}-${15} \
	--partition m100_usr_prod \
	--time 24:00:00 --nodes 1 --cpus-per-task 8 --gres gpu:1 --mem 60GB \
	--mail-user alessio.burrello@polito.it --mail-type ALL \
	--output $HOME/Alessio/bioformer/log/%x.log \
	--error $HOME/Alessio/bioformer/err_log/%x.log \
	$HOME/Alessio/bioformer/run_networks.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}
