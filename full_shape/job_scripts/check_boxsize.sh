#!/bin/bash
# salloc -N 1 -C "gpu&hbm80g" -t 00:10:00 --gpus 4 --qos interactive --account desi_g
# salloc -N 1 -C "gpu&hbm80g" -t 00:20:00 --gpus 4 --gpus-per-node=4 --qos interactive --account desi_g
# source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test
# bash check_boxsize.sh QSO 451
# bash check_boxsize.sh LRG 451
# bash check_boxsize.sh ELG_LOPnotqso 451

set -e
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
CAT_DIR=/dvs_ro/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/holi_v1/
MEAS_DIR=$SCRATCH/cai-dr2-benchmarks/boxsize_checks/
CODE=../compute_fiducial_stats.py

TRACER=$1
imock=$2

JOB_FLAGS="-N 1 -n 4"
echo $JOB_FLAGS

# settings tested for Y3
NRAN_QSO=10  # 4 randoms gives x55, 5 gives x69, 3 gives x41.4
NRAN_LRG=10  # 9 randoms gives x50.5, 11 gives x61.9, 8 gives x45.0
NRAN_ELG=10 # 13 randoms gives x51.4, 16 gives x63.2, 11 gives x43.5

CELLSIZE=7.8 # Mpc/h

COMMON_FLAGS="--stats mesh2_spectrum --region NGC SGC --cellsize $CELLSIZE --cat_dir $CAT_DIR/altmtl$imock/loa-v1/mock$imock/LSScats/ --meas_dir $MEAS_DIR/mock$imock/"
# COMMON_FLAGS="--todo mesh3_spectrum_sugiyama combine --region NGC SGC --cellsize $CELLSIZE"
if [ $TRACER == 'ELG_LOPnotqso' ]; then
    TRACER_FLAGS="--tracer $TRACER --nran $NRAN_ELG --zrange 0.8 1.1 1.1 1.6 --weight_type default_FKP"
    list=(6000 7000 8000 9000 10000)
fi

if [ $TRACER == 'LRG' ]; then
    TRACER_FLAGS="--tracer $TRACER --nran $NRAN_LRG --zrange 0.4 0.6 0.6 0.8 0.8 1.1 --weight_type default_FKP"
    list=(5000 6000 7000 8000 9000 10000)
fi

if [ $TRACER == 'QSO' ]; then
    TRACER_FLAGS="--tracer $TRACER --nran $NRAN_QSO --zrange 0.8 2.1 --weight_type default_FKP"
    list=(7000 8000 9000 10000)
fi
echo $COMMON_FLAGS $TRACER_FLAGS

for boxsize in "${list[@]}"; do
    echo "Working on $TRACER for boxsize $boxsize and saving to $MEAS_DIR"
    srun $JOB_FLAGS python $CODE $COMMON_FLAGS $TRACER_FLAGS --boxsize $boxsize --meas_extra "boxsize$boxsize"
done
