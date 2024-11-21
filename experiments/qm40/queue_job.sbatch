#!/usr/bin/env zsh
#SBATCH -J qm40
#SBATCH -A ekraka_drugdesign_0001
#SBATCH -o output/qm40_schnet_%j.out
#SBATCH -c 128 --mem=16G     
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --time=0-16:00:00 
#SBATCH --partition=batch

framework="jax"

if [ "$framework" = "jax" ]; then
    script_file="train_qm40_jax.py"
else
    script_file="qm40_nn_conv.py"
fi

save_mnt="/lustre/smuexa01/client/users/ejlaird/smartcadd/qm40_results/${SLURM_JOB_ID}"
mkdir -p ${save_mnt}/QM40/raw

cp /work/users/ejlaird/data/QM40/*.csv ${save_mnt}/QM40/raw


target=0
dim=64
batch_size=32
lr=0.0001
model="SchNet"
max_nodes=92
max_edges=8372
layers=6
epochs=1000
r_cutoff=20.0
l2_lambda=1e-5

pip_install="pip install --upgrade -q 'jax[cuda12_local]' jraph dm-haiku optax torch-jax-interop -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
python_script="python /work_dir/experiments/qm40/"${script_file}" --epochs ${epochs} --target ${target} --dim ${dim} --batch_size ${batch_size} --lr ${lr} --model ${model} --max_nodes ${max_nodes} --max_edges ${max_edges} --layers ${layers} --l2_lambda ${l2_lambda} --r_cutoff ${r_cutoff}"

srun\
    --no-container-entrypoint\
    --container-image /work/group/humingamelab/sqsh_images/nvidia-pyg.sqsh\
    --container-mounts="${HOME}"/Projects/smartcadd2:/work_dir,"${save_mnt}":/data,"${save_mnt}":/artifacts\
    --container-workdir /work_dir\
    bash -c "${pip_install}; ${python_script}"
