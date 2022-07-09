ml gcc/7.1.0
ml python3/3.6.1 cuda/9.0 cudnn/7.4.2
python3 run.py --RUN='train' --VERSION='0107_dp1_cr0_lp1_k0p01_etam294_sdim2_arm_wt_relu_cha32' --SPLIT='train' --DP_TYPE=1 --CONCRETE=0 --LEARNPRIOR=1 --DP_K=0.01 --DP_ETA=-294 --ARM=1 --GPU='0'