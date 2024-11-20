#!/usr/bin/env zsh
#SBATCH -J qm9
#SBATCH -A ekraka_drugdesign_0001
#SBATCH -o output/qm9_schnet_%j.out
#SBATCH -c 16 --mem=16G     
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --time=0-04:00:00
#SBATCH --mail-user ejlaird@smu.edu
#SBATCH --mail-type=ALL     
#SBATCH --partition=batch

framework="jax"

if [ "$framework" = "jax" ]; then
    script_file="train_qm9_jax.py"
else
    script_file="qm9_nn_conv.py"
fi


target=7
dim=64
batch_size=128
lr=0.0005
data_dir="/data"
save_dir="/artifacts/qm9"
model="SchNet"
max_nodes=29
max_edges=812
layers=6
epochs=1000
r_cutoff=20.0

pip_install="pip install --upgrade -q 'jax[cuda12_local]' jraph dm-haiku optax torch-jax-interop -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
python_script="python /work_dir/experiments/qm9/"${script_file}" --target ${target} --dim ${dim} --batch_size ${batch_size} --lr ${lr} --data_dir ${data_dir} --save_dir ${save_dir} --model ${model} --max_nodes ${max_nodes} --max_edges ${max_edges} --layers ${layers}"

srun\
    --no-container-entrypoint\
    --container-image /work/group/humingamelab/sqsh_images/nvidia-pyg.sqsh\
    --container-mounts="${HOME}"/Projects/smartcadd2:/work_dir,/work/users/ejlaird/data/QM9/:/data,/work/users/ejlaird/smartcadd2_results:/artifacts\
    --container-workdir /work_dir\
    bash -c "${pip_install}; ${python_script}"
