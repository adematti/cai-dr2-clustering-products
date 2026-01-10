#!/bin/bash
# salloc -N 1 -C gpu -t 00:10:00 --gpus 4 --qos interactive --account desi_g
# salloc -N 1 -C "gpu&hbm80g" -t 00:10:00 --gpus 4 --qos interactive --account desi_g
# source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test
# bash run_test.sh

set -e
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test
# DA2/LSS/loa-v1/LSScats/v2

INPUT_DIR=/dvs_ro/cfs/cdirs/desi/survey/catalogs//Y1/LSS/iron/LSScats/v1.5/
OUTPUT_DIR=$PSCRATCH/checks2/

ELG=ELG_LOPnotqso
LRG=LRG
QSO=QSO

BOX=10000
PK_NRAN=10

# flags="--basedir $INPUT_DIR --outdir $OUTPUT_DIR --boxsize $BOX --cellsize 12 --nran $PK_NRAN --weight_type default_FKP"
# srun -N 1 -n 4 python jax-bkrun.py $flags --tracer $QSO --zrange 0.8 2.1 --todo mesh3_spectrum_sugiyama --region NGC SGC

flags="--basedir $INPUT_DIR --outdir $OUTPUT_DIR --boxsize $BOX --cellsize 10 --nran $PK_NRAN --weight_type default_FKP"
#srun -N 1 -n 4 python jax-pkrun.py $flags --tracer $QSO --zrange 0.8 2.1 --todo mesh2_spectrum combine --region NGC SGC
srun -N 1 -n 4 python jax-pkrun.py $flags --tracer $LRG --zrange 0.4 0.6 0.6 0.8 0.8 1.1 --todo mesh2_spectrum combine --region NGC SGC

#flags="--basedir $INPUT_DIR --outdir $OUTPUT_DIR --region NGC SGC --boxsize $BOX --cellsize 10 --nran $PK_NRAN"
#flags="--basedir $INPUT_DIR --outdir $OUTPUT_DIR --boxsize $BOX --cellsize 10 --nran $PK_NRAN --weight_type default_FKP"
#srun -N 1 -n 4 python jax-pkrun.py $flags --tracer $QSO --zrange 0.8 2.1 --todo mesh2_spectrum combine --region NGCnoN SGCnoDES
# srun -N 1 -n 4 python jax-pkrun.py $flags --tracer $QSO --zrange 0.8 2.1 --todo combine --region NGC SGC
# srun -N 1 -n 4 python jax-pkrun.py $flags --tracer $QSO --zrange 0.8 2.1 --todo combine --region NGCnoN SGC
#srun -N 1 -n 4 python jax-pkrun.py $flags --tracer $QSO --zrange 0.8 2.1 --todo combine --region NGC SGCnoDES