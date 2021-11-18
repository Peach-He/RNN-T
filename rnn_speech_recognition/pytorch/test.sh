# source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
# export LD_LIBRARY_PATH=/opt/intel/oneapi/intelpython/python3.7/envs/pytorch/lib/python3.7/site-packages/torch/lib/

export LD_LIBRARY_PATH=/opt/intel/oneapi/intelpython/python3.7/envs/pytorch_mlperf/lib/python3.7/site-packages/torch/lib/
source /opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/.local/env/setvars.sh
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"
# Example:
# Run 2 processes on 2 sockets. (28 cores/socket, 4 cores for CCL, 24 cores for computation)
#
# CCL_WORKER_COUNT means per instance threads used by CCL.
# CCL_WORKER_COUNT, CCL_WORKER_AFFINITY and I_MPI_PIN_DOMAIN should be consistent.
export CCL_WORKER_COUNT=4
export CCL_WORKER_AFFINITY="0,1,2,3,28,29,31,32"

mpiexec.hydra -np 2 -ppn 2 -l -genv I_MPI_PIN_DOMAIN=[0x0000000FFFFFF0,0xFFFFFF00000000] \
              -genv KMP_BLOCKTIME=1 -genv KMP_AFFINITY=granularity=fine,compact,1,0      \
              -genv OMP_NUM_THREADS=16 /opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python -u test.py