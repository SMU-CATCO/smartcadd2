#!/usr/bin/env zsh
#SBATCH -J qm9
#SBATCH -o output/qm9_%j.out
#SBATCH -c 128 --mem=120G     
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --time=1-23:59:00
#SBATCH --mail-user ejlaird@smu.edu
#SBATCH --mail-type=ALL     
#SBATCH --exclude=bcm-dgxa100-0014

framework="jax"

if [ "$framework" = "jax" ]; then
    script_file="train_qm9_jax.py"
else
    script_file="qm9_nn_conv.py"
fi

CUDA_HOME="/usr/local/cuda-12"

pip_install="pip install --upgrade -q 'jax[cuda12_local]' jraph dm-haiku optax torch-jax-interop -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
python_script="python /work_dir/experiments/qm9/"${script_file}" --target 0 --dim 256 --batch_size 128 --data_dir /data --save_dir /artifacts/qm9 --model GAT --heads 4 --concat True --max_nodes 29 --max_edges 812"
# set_ld_library_path="export LD_LIBRARY_PATH='${CUDA_HOME}/lib64:${CUDA_HOME}/compat/lib.real:$LD_LIBRARY_PATH'"


srun\
    --no-container-entrypoint\
    --container-image /work/group/humingamelab/sqsh_images/nvidia-pyg.sqsh\
    --container-mounts="${HOME}"/Projects/smartcadd2:/work_dir,/work/users/ejlaird/data/QM9/:/data,/work/users/ejlaird/smartcadd2_results:/artifacts\
    --container-workdir /work_dir\
    bash -c "${pip_install}; ${python_script}"
