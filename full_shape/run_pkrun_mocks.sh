#!/bin/bash
# salloc -N 1 -C "gpu&hbm80g" -t 00:10:00 --gpus 4 --qos interactive --account desi_g
# salloc -N 1 -C "gpu&hbm80g" -t 00:20:00 --gpus 4 --gpus-per-node=4 --qos interactive --account desi_g
# salloc -N 4 -C "gpu&hbm80g" -t 00:20:00 --gpus 16 --gpus-per-node=4 --qos interactive --account desi_g
# source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test
# bash run_pkrun_mocks.sh QSO

set -e
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test
#/dvs_ro/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/holi_v1/altmtl451/loa-v1/mock451/LSScats/
#DA2/mocks/SecondGenMocks/AbacusSummit/desipipe/v4_1/altmtl/2pt/mock0/
#/dvs_ro/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe/holi_v1/altmtl/2pt/mock{i}/
INPUT_DIR=/dvs_ro/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/holi_v1/
OUTPUT_DIR=$PSCRATCH/checks4/checks1/
CODE=jax-pkrun.py

TRACER=$1

#list=({451..500} {601..650})
list=({451..451})
length=${#list[@]}

N=$SLURM_NNODES
ntasks=4
nmax=$((N * 1))
gpus_per_task=$((N * 4 / ntasks / nmax)) 
cpu=$((2 * 64 / ntasks))
JOB_FLAGS="-N 1 -n $ntasks" 
# -c $cpu --gpus-per-task $gpus_per_task"
echo $JOB_FLAGS

BOXSIZE_LRG=7000
BOXSIZE_ELG=9000
BOXSIZE_QSO=10000

NRAN_ELG=10
NRAN_LRG=10
NRAN_QSO=10


COMMON_FLAGS="--todo mesh2_spectrum combine --region NGC SGC --cellsize 10" 
if [ $TRACER == 'ELG_LOP' ]; then
    TRACER_FLAGS="--tracer $TRACER --boxsize $BOXSIZE_ELG --nran $NRAN_ELG --zrange 0.8 1.1 1.1 1.6 --weight_type default_FKP"   
fi

if [ $TRACER == 'LRG' ]; then
    TRACER_FLAGS="--tracer $TRACER --boxsize $BOXSIZE_LRG --nran $NRAN_LRG --zrange 0.4 0.6 0.6 0.8 0.8 1.1 --weight_type default_FKP"   
fi

if [ $TRACER == 'QSO' ]; then
    TRACER_FLAGS="--tracer $TRACER --boxsize $BOXSIZE_QSO --nran $NRAN_QSO --zrange 0.8 2.1 --weight_type default_FKP"
fi
echo $PKFLAGS $TRACER_FLAGS

incomplete_chunk=()
echo $nmax
for ((i=0; i<length; i+=$nmax)); do
    remaining=$((length - i))
    if (( remaining >= nmax )); then
        echo "Working on $TRACER for mocks ${list[@]:i:nmax} and saving to $OUTPUT_DIR"
        for ((j=0; j<nmax; j++)); do
            mocki="${list[i+j]}"
            srun $JOB_FLAGS python $CODE $COMMON_FLAGS $TRACER_FLAGS --basedir $INPUT_DIR/altmtl$mocki/loa-v1/mock$mocki/LSScats/ --outdir $OUTPUT_DIR/mock$mocki/ &
        done
        wait
    else
        # Store remaining mocks
        incomplete_chunk+=("${list[@]:i}")
    fi
done

# Later use of incomplete chunk
if (( ${#incomplete_chunk[@]} > 0 )); then
    echo "Processing stored incomplete chunk:"
    for mocki in "${incomplete_chunk[@]}"; do
        echo "Working on mock $mocki"
        srun $JOB_FLAGS python $CODE $COMMON_FLAGS $TRACER_FLAGS --basedir $INPUT_DIR/altmtl$mocki/loa-v1/mock$mocki/LSScats/ --outdir $OUTPUT_DIR/mock$mocki/ &
    done
    wait
fi