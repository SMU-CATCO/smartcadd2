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


srun\
    --no-container-entrypoint\
    --container-image /work/group/humingamelab/sqsh_images/nvidia-pyg.sqsh\
    --container-mounts="${HOME}"/Projects/smartcadd2:/work_dir,/work/users/ejlaird/data/QM40/:/data,/work/users/ejlaird/smartcadd2_results:/artifacts\
    --container-workdir /work_dir\
    bash -c "python /work_dir/experiments/qm9/qm9_nn_conv.py --target 7 --dim 64 --batch_size 128 --data_dir /data --save_dir /artifacts/qm9 --model GAT --heads 8 --concat True"
