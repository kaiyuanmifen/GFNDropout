#!/bin/bash
#onlocal computer



#compute canada
ssh dianbo@cedar.computecanada.ca

#cd /home/dianbo/projects/def-bengioy/Dianbo/Disentanglement

cd /home/dianbo/scratch/Disentanglement

module load python/3.7
module load cuda/11.0
source /home/dianbo/projects/def-bengioy/Dianbo/Disentanglement/GNN/bin/activate

scp -r Stage26_objTIM/ dianbo@cedar.computecanada.ca:/home/dianbo/scratch/Disentanglement/

scp -r dianbo@cedar.computecanada.ca:/home/dianbo/Disentanglement/ExampleImages ./

salloc --time=0-05:00 --ntasks=1 --gres=gpu:1 --account=rrg-bengioy-ad --mem=25G 

salloc --time=0-03:00 --ntasks=1 --gres=gpu:1 --account=def-bengioy --mem=20G 





module load python/3.7

virtualenv --no-download  GNN

source GNN/bin/activate

pip install --no-index --upgrade pip

pip install --no-index torch==1.8.1

pip install --no-index torchvision torchtext torchaudio

pip install gym==0.12.0

pip install --no-index atari-py

#pip install --no-index scikit-image

pip install --no-index matplotlib
pip install --no-index h5py
pip install scikit-image==0.15.0
##On windows

C:\Users\kaiyu\AppData\Local\Programs\Python\Python36\Scripts\>GNN\Scripts\activate.bat


GNN\Scripts\deactivate.bat








##Ubuntu
#conda create -n GFlownets python=3.6 anaconda
# conda create -n VirtualTool python=3.7 anaconda

cd /mnt/c/Users/kaiyu/Google\ Drive/research/MILA/GFlownets


conda activate GFlownets

conda activate visualprompting


conda activate NPS


conda activate GNN


conda activate TIM



conda create -n phyre python=3.6 anaconda
conda activate phyre
pip install phyre
#conda create -n GNN python=3.7 anaconda
#source MiniGrid_RIM_Env_New_python3.7/bin/activate
conda create -n TextProcessing python=3.7 anaconda










#on Mila cluster
ssh dianbo.liu@login.server.mila.quebec -p 2222

cd /home/mila/a/aniket.didolkar/modular_central/bb_anirudh/dataset_scripts/
https://docs.mila.quebec

scp -P 2222 -r StageTwo_10_joint dianbo.liu@login.server.mila.quebec:/home/mila/d/dianbo.liu/scratch/GFlownets/Dropout/

scp -P 2222 -r *.py dianbo.liu@login.server.mila.quebec:/home/mila/d/dianbo.liu/scratch/GFlownets/Dropout/StageTwo_10_joint/BayesianGlowout/

scp -P 2222 -r dianbo.liu@login.server.mila.quebec:/home/mila/d/dianbo.liu/scratch/GFlownets/Dropout/StageTwo_10_joint/BayesianGlowout/logs ./

scp -P 2222 -r dianbo.liu@login.server.mila.quebec:/home/mila/d/dianbo.liu/scratch/GFlownets/Dropout/StageTwo_10_joint/BayesianGlowout/checkpoints ./


scp -P 2222 -r dianbo.liu@login.server.mila.quebec:/home/mila/d/dianbo.liu/scratch/GFlownets/Dropout/Stage31VAGFNCNN/log ./




scp -P 2222 -r StageE38_probing dianbo.liu@login.server.mila.quebec:/home/mila/d/dianbo.liu/scratch/GFlownets/Surprise/
scp -P 2222 *.sh dianbo.liu@login.server.mila.quebec:/home/mila/d/dianbo.liu/scratch/GFlownets/Surprise/StageE9/C-SWM/
scp -P 2222 *.py dianbo.liu@login.server.mila.quebec:/home/mila/d/dianbo.liu/scratch/GFlownets/Surprise/StageE9/C-SWM/



####Dropout

scp -P 2222 -r  Stage52_solvingcollapse dianbo.liu@login.server.mila.quebec:/home/mila/d/dianbo.liu/scratch/GFlownets/Dropout/
scp -P 2222 *.sh dianbo.liu@login.server.mila.quebec:/home/mila/d/dianbo.liu/scratch/GFlownets/Dropout/Stage20_clean/
scp -P 2222 *.py dianbo.liu@login.server.mila.quebec:/home/mila/d/dianbo.liu/scratch/GFlownets/Dropout/Stage20_clean/


scp -P 2222 -r dianbo.liu@login.server.mila.quebec:/home/mila/d/dianbo.liu/scratch/GFlownets/Dropout/Stage52_solvingcollapse/Pretrained_model/images ./

scp -P 2222 -r dianbo.liu@login.server.mila.quebec:/home/mila/d/dianbo.liu/scratch/GFlownets/Dropout/Stage50_twolayermasks/Results ./

tensorboard --logdir='log'


python main_train.py \
--backbone-depth 20 \
--data-dir ../../../data/



salloc --gres=gpu:1 --mem=32G -t 4:00:00 --partition=unkillable

###game envirmnemt


from gym import envs
print(envs.registry.all())


xvfb-run -s "-screen 0 1400x900x24" python ViewEnvs.py
. ~/.bashrc
https://dibranmulder.github.io/2019/09/06/Running-an-OpenAI-Gym-on-Windows-with-WSL/

#source .bashrc

#module avail
salloc --gres=gpu:1 -c 1 --mem 35G



salloc --gres=gpu:1 --mem=32G -t 4:00:00 --partition=unkillable

#module load python/3.6
module load anaconda/3
#module load cuda/10.1
module load cuda/11.1

conda activate GFlownets

#conda create -n GFlownets python=3.6 anaconda
#conda create -n Dropout python=3.6 anaconda
conda activate GNN

conda activate MARL

conda activate GNN


conda activate GFlownets
conda env export > environment.yml


# module load python/3.6
# module load anaconda/3
# module load cuda/10.1
# #conda create -n NPS python=3.6 anaconda
# conda activate NPS

pip3 freeze > requirements.txt 
#conda create -n virtualtools python=3.6 anaconda

#conda create -n phyre python=3.6
conda activate phyre
#pip install phyre





module load cuda/10.1
module load python/3.7
module load pytorch/1.6
#cd /home/mila/d/dianbo.liu/RIM
source /home/mila/d/dianbo.liu/RIM/torch1/bin/activate

#virtualenv torch1
#cd /home/mila/d/dianbo.liu/RIM
#source torch1/bin/activate
#pip install -r requirement.txt

#srun --pty -p interactive --mem 20G -t 0-06:00 /bin/bash

cd /home/mila/a/aniket.didolkar/modular_central/bb_anirudh/dataset_scripts/



git add .

git commit -m "DL"

git push origin main



beegfs-ctl --cfgFile=/etc/beegfs/home.d/beegfs-client.conf --getquota --uid dianbo.liu

https://docs.mila.quebec/Information.html#storage

git pull 

ssh dianbo.liu@login.server.mila.quebec -p 2222








python POO_test_Cleaner.py