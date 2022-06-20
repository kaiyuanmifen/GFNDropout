import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Find CIFAR10, MNIST and SVHN
# Take second part of _
# Get the performances.csv
# Get the losses.csv
# Make performance df
# Make loss df
# Plot time series with sns

FOLDER = "Results_Final_18_600_2"
IMAGE_FOLDER = f"image_{FOLDER}"
os.makedirs(IMAGE_FOLDER,exist_ok=True)

files = [f.name for f in os.scandir(FOLDER)]
files = [f for f in files if f.endswith('.csv')] # Get only CSV files

for image_type in ['CIFAR10','MNIST','SVHN']:
    for N_units in [1024]:
        for p in [0.5]:
            for beta in [1.0]:
                for OODReward in [2]:
                    for data_ratio in [1.0]:
                        for seed in [1]:
                            loss_dfs=[]
                            acc_dfs=[]
                            for exp_type in ["RESNET_nodropout","RESNET_Standout","RESNET_dropout","RESNET_SVD", "RESNET_GFFN"]:

                                Task_name=exp_type+"_"+image_type+"_"+str(N_units)+"_"+str(p)+"_"+str(beta)+"_"+str(OODReward)+"_"+str(data_ratio)+"_"+str(seed)
                                losses_csv = os.path.join(FOLDER,Task_name+'_losses.csv')
                                acc_csv = os.path.join(FOLDER,Task_name+'_performance.csv')

                                try:
                                    #losses_df = pd.read_csv(losses_csv,index_col=False).drop(columns=['Unnamed: 0'])
                                    #acc_df = pd.read_csv(acc_csv,index_col=False).drop(columns=['Unnamed: 0'])
                                    losses_df = pd.read_csv(losses_csv,index_col=False)
                                    acc_df = pd.read_csv(acc_csv,index_col=False)

                                    losses_df['Method'] = [exp_type for i in range(len(losses_df))]
                                    acc_df['Method'] = [exp_type for i in range(len(acc_df))]

                                    losses_df['train_loss_log'] = losses_df['train_loss'].apply(np.log)
                                    
                                    losses_df['Step'] = [str(i) for i in range(len(losses_df))]
                                    acc_df['Step'] = [str(i) for i in range(len(acc_df))]

                                    loss_dfs.append(losses_df)
                                    acc_dfs.append(acc_df) 
                                except Exception as e:
                                    continue    

                            all_loss_df = pd.concat(loss_dfs,ignore_index=True)
                            all_acc_df = pd.concat(acc_dfs,ignore_index=True)


                            plt.figure()
                            ax = sns.lineplot(x="Step", y="train_loss_log", hue="Method",data=all_loss_df)
                            ax.set_title(image_type)

                            ax.set_xticks(all_loss_df['Step'][::100])        
                            plt.savefig(f"{IMAGE_FOLDER}/{Task_name}_loss.png")    

                            # for acc_OOD
                            plt.figure()
                            ax =sns.lineplot(x="Step", y="test_acc_OOD", hue="Method",data=all_acc_df)
                            ax.set_xticks(all_acc_df['Step'][::100])        

                            ax.set_title(image_type)
                            plt.savefig(f'{IMAGE_FOLDER}/{Task_name}_acc_OOD.png')            

                            # for acc
                            plt.figure()
                            ax = sns.lineplot(x="Step", y="test_acc", hue="Method",data=all_acc_df)
                            ax.set_xticks(all_acc_df['Step'][::100])        

                            ax.set_title(image_type)
                            plt.savefig(f'{IMAGE_FOLDER}/{Task_name}_test_acc.png') 

                            # for CIFAR_10C_acc
                            if image_type=="CIFAR10":
                                try:
                                    plt.figure()
                                    ax = sns.lineplot(x="Step", y="CIFAR_10C_acc", hue="Method",data=all_acc_df)
                                    ax.set_xticks(all_acc_df['Step'][::100])        

                                    ax.set_title(image_type)
                                    plt.savefig(f'{IMAGE_FOLDER}/{Task_name}_CIFAR10C_acc.png')
                                except Exception as err:
                                    continue     