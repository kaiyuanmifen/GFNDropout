ml gcc/7.1.0
ml python3/3.6.1 cuda/9.0 cudnn/7.4.2
python3 run.py --RUN='val' --CKPT_V='dp0_cr1_lp1_k0p01_etam2p19' --SPLIT='train' --DP_TYPE=0 --CONCRETE=1 --LEARNPRIOR=1 --DP_K=0.01 --DP_ETA=-219 --CKPT_E=13 --BS=128 --UNCERTAINTY_SAMPLE=5