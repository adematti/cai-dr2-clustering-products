#!/bin/bash
#SBATCH --account desi_g
#SBATCH -C gpu&hbm80g
#SBATCH -N 1
#SBATCH --gpus 4
#SBATCH -t 0:20:00
#SBATCH -q regular
#SBATCH -J mocks_pks
#SBATCH -L SCRATCH
#SBATCH -o slurm_outputs/holi_mocks_pks_%A/mock%a.log
#SBATCH --array=451-500,601-650

set -e
# Timer initialisation:
SECONDS=0

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test

mocki=$SLURM_ARRAY_TASK_ID
INPUT_DIR=/dvs_ro/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/holi_v1/altmtl$mocki/loa-v1/mock$mocki/LSScats/
OUTPUT_DIR=/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe/holi_v1/altmtl/2pt/mock$mocki/pk/

CODE=../jax-pkrun.py
echo $INPUT_DIR
echo $OUTPUT_DIR

BOXSIZE_LRG=7000
BOXSIZE_ELG=9000
BOXSIZE_QSO=10000

NRAN_ELG=10
NRAN_LRG=10
NRAN_QSO=10

JOB_FLAGS="-N 1 -n 4" 
COMMON_FLAGS="--todo mesh2_spectrum combine --region NGC SGC --cellsize 10 --basedir $INPUT_DIR --outdir $OUTPUT_DIR"

ELG_FLAGS="--tracer ELG_LOPnotqso --boxsize $BOXSIZE_ELG --nran $NRAN_ELG --zrange 0.8 1.1 1.1 1.6 --weight_type default_FKP"   
LRG_FLAGS="--tracer LRG --boxsize $BOXSIZE_LRG --nran $NRAN_LRG --zrange 0.4 0.6 0.6 0.8 0.8 1.1 --weight_type default_FKP"      
QSO_FLAGS="--tracer QSO --boxsize $BOXSIZE_QSO --nran $NRAN_QSO --zrange 0.8 2.1 --weight_type default_FKP"

srun $JOB_FLAGS python $CODE $ELG_FLAGS $COMMON_FLAGS
srun $JOB_FLAGS python $CODE $LRG_FLAGS $COMMON_FLAGS
srun $JOB_FLAGS python $CODE $QSO_FLAGS $COMMON_FLAGS

echo " "
if (( $SECONDS > 3600 )); then
    let "hours=SECONDS/3600"
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $hours hour(s), $minutes minute(s) and $seconds second(s)"
elif (( $SECONDS > 60 )); then
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $minutes minute(s) and $seconds second(s)"
else
    echo "Completed in $SECONDS seconds"
fi
echo
