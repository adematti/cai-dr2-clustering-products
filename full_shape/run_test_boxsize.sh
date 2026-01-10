#!/bin/bash
# salloc -N 1 -C "gpu&hbm80g" -t 00:10:00 --gpus 4 --qos interactive --account desi_g
# salloc -N 1 -C "gpu&hbm80g" -t 00:20:00 --gpus 4 --gpus-per-node=4 --qos interactive --account desi_g
# source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test
# bash run_test_boxsize.sh QSO 451
# bash run_test_boxsize.sh LRG 451
# bash run_test_boxsize.sh ELG_LOPnotqso 451

set -e
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
INPUT_DIR=/dvs_ro/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/holi_v1/
OUTPUT_DIR=$PSCRATCH/cai-dr2-benchmarks/boxsize_checks/
# CODE=jax-bkrun.py
CODE=jax-pkrun.py

TRACER=$1
mocki=$2 


JOB_FLAGS="-N 1 -n 4" 
echo $JOB_FLAGS

# settings tested for Y3
NRAN_QSO=10  # 4 randoms gives x55, 5 gives x69, 3 gives x41.4
NRAN_LRG=10  # 9 randoms gives x50.5, 11 gives x61.9, 8 gives x45.0
NRAN_ELG=10 # 13 randoms gives x51.4, 16 gives x63.2, 11 gives x43.5

BOXSIZE_QSO=10000
BOXSIZE_LRG=10000
BOXSIZE_ELG=10000

CELLSIZE=7.8 # Mpc/h

COMMON_FLAGS="--todo mesh2_spectrum combine --region NGC SGC --cellsize $CELLSIZE --basedir $INPUT_DIR/altmtl$mocki/loa-v1/mock$mocki/LSScats/ --outdir $OUTPUT_DIR/mock$mocki/" 
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
    echo "Working on $TRACER for boxsize $boxsize and saving to $OUTPUT_DIR"
    srun $JOB_FLAGS python $CODE $COMMON_FLAGS $TRACER_FLAGS --boxsize $boxsize
done

