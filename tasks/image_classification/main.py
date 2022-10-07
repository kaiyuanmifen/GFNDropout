import torch
from torchnet import meter
from torch import nn
from tqdm import tqdm
import data.dataset as dataset
import models
from utils.visualization import Visualizer
import numpy as np
import time
from time import localtime
from config import opt
from collections import OrderedDict
import os
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import random
import scipy.stats as sts
#updating
import pandas as pd
from six.moves import cPickle
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
from ece import ECELoss
from random import randrange
from sklearn.metrics import f1_score, recall_score, precision_score, balanced_accuracy_score

import warnings
warnings.filterwarnings("ignore")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_seed_2(seed):
    np.random.seed(seed)


#device = None
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True
#current time updates
current_time = time.strftime('%Y%m%d%H%M%S', localtime())
print(current_time)
vis = None


def image_augmentation(inputs):
    inputs = inputs.squeeze()
    #im = transforms.ToPILImage()(inputs)
    #im=inputs
    if len(inputs.shape)==2:# if single color image
        bwimage=True
        inputs=inputs.unsqueeze(0).repeat(3,1,1)
    else:
        bwimage=False
    grouped_z1_pil = transforms.RandomRotation(degrees=(0,360))(inputs)
    #grouped_z1_pil = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(5.0))(inputs)
    
    #grouped_z1_pil = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.5))(im)
    #output_tensor = torchvision.transforms.ToTensor()(grouped_z1_pil)
    #output_tensor = torchvision.transforms.ToTensor()(im)
    output_tensor =grouped_z1_pil
    if bwimage:
        output_tensor=output_tensor[0,:,:]
    return output_tensor


def train(active_learning=False, **kwargs):
    global device
    if opt.seed is not None:
        setup_seed(opt.seed)
    if opt.valtestseed is not None:
        setup_seed_2(opt.valtestseed)
    print("user set commands:")
    print(kwargs)
    config_str = opt.parse(kwargs)
    #device = torch.device("cuda" if opt.use_gpu else "cpu")
    device==torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_val_loss=1e9
    #format
    # vis = Visualizer(opt.log_dir, opt.model+opt.model_name, current_time, opt.title_note)
    # log all configs
    # vis.log('config', config_str)

    # load data set
    if active_learning == False:
    	train_loader, val_loader,test_loader, num_classes = getattr(dataset, opt.dataset)(opt.batch_size )
    else:
    	train_loader, val_loader,test_loader, num_classes = kwargs['new_data']
    # load model
    print("***********************opt.model****")
    print(opt.model)
    model = getattr(models, opt.model)(lambas=opt.lambas, num_classes=num_classes,
         weight_decay=opt.weight_decay,opt=opt).to(
        device)

    model.train()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print('************total params size: ', pytorch_total_params)
    
    #resume the model
    if opt.start_model is not None:
        #model.load_state_dict(torch.load(os.path.join(opt.start_model, '100.model'))
        model.load_state_dict(torch.load(os.path.join(opt.start_model, 'best.model'), map_location=device))
        print("loading from existing model:",os.path.join(opt.start_model, 'best.model'))
    if opt.gpus > 1:
        model = nn.DataParallel(model)



    #updating
    histories = {}

    if opt.start_from is not None:
    # open old infos and check if models are compatible
        if os.path.isfile(os.path.join(opt.start_from, 'histories_' + '.pkl')):
            with open(os.path.join(opt.start_from, 'histories_' + '.pkl'), 'rb') as f:
                histories = cPickle.load(f)

    #dp_history = histories.get('dp_history', {})
    target_history = histories.get('target_history', {})
    input__history = histories.get('input__hisotry',{})
    val_accuracy_history = histories.get('val_accuracy_hisotry', {})
    first_order = histories.get('first_order_history', np.zeros(1))
    second_order = histories.get('second_order_history', np.zeros(1))
    first_order = torch.from_numpy(first_order).float().to(device)
    second_order = torch.from_numpy(second_order).float().to(device)
    variance_history = histories.get('variance_history', {})





    # define loss function
    def criterion(output, target_var):

        loss = nn.CrossEntropyLoss().to(device)(output, target_var.long())
        if opt.GFN_dropout==False:
            reg_loss = model.regularization() if opt.gpus <= 1 else model.module.regularization()
            #total_loss = (loss + reg_loss.type_as(loss)).to(device)
            total_loss = (loss + reg_loss).to(device)
            return total_loss
        else:
            return loss


    # load optimizer and scheduler
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters() if opt.gpus <= 1 else model.module.parameters(), opt.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=opt.lr_decay, patience=15)
        # scheduler = None
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.schedule_milestone,gamma=opt.lr_decay)
        print('Optimizer: Adam, lr={}'.format(opt.lr))
    elif opt.optimizer == 'momentum':
        optimizer = torch.optim.SGD(model.parameters() if opt.gpus <= 1
                                    else model.module.parameters(), opt.lr, momentum=opt.momentum, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.schedule_milestone,gamma=opt.lr_decay)
        print('Optimizer: Momentum, lr={}, momentum'.format(opt.lr, opt.momentum))
    else:
        print('No optimizer')
        return


    # resume the model
    if opt.start_optim is not None:
        #optimizer.load_state_dict(torch.load(os.path.join(opt.start_optim, '100.optim')))
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_optim, str(opt.start_epoch)+'.optim'), map_location=device))


    loss_meter = meter.AverageValueMeter()
    if opt.GFN_dropout==True:
        accuracy_meter = meter.AverageValueMeter()
        
        GFNloss_unconditional_meter=meter.AverageValueMeter()
        LogZ_unconditional_meter=meter.AverageValueMeter()
        LogPF_qz_meter=meter.AverageValueMeter()

        GFNloss_conditional_meter=meter.AverageValueMeter()
        LogZ_conditional_meter=meter.AverageValueMeter()
        LogPF_qzxy_meter=meter.AverageValueMeter()

        LogPF_BNN_meter=meter.AverageValueMeter()

        actual_dropout_rate_meter=meter.AverageValueMeter()

        COR_qz_meter=meter.AverageValueMeter()

        COR_qzxy_meter=meter.AverageValueMeter()

        Log_pz_meter=meter.AverageValueMeter()

        Log_pzx_meter=meter.AverageValueMeter()


    else:
        accuracy_meter = meter.ClassErrorMeter(accuracy=True)
    
    # create checkpoints dir
    #format
    #TODO: neet to change
    if opt.GFN_dropout:
    	try:
    		directory = '{}/{}_{}_{}'.format(opt.checkpoints_dir, opt.model + opt.model_name+"_"+opt.mask+"_"+str(opt.BNN), current_time, str(opt.al_round))
    	except:
    		directory = '{}/{}_{}'.format(opt.checkpoints_dir, opt.model + opt.model_name+"_"+opt.mask+"_"+str(opt.BNN), current_time)
    else:
    	try:
    		directory = '{}/{}_{}_{}'.format(opt.checkpoints_dir, opt.model + opt.model_name, current_time, str(opt.al_round))
    	except:
    		directory = '{}/{}_{}'.format(opt.checkpoints_dir, opt.model + opt.model_name, current_time)
    
    #directory = '{}/{}'.format(opt.checkpoints_dir, opt.model + opt.model_name)
    
    print("directory for saving")
    print(directory)
    #directory = '.'
    if not os.path.exists(directory):
        os.makedirs(directory)
    total_steps = 0
    for epoch in range(opt.start_epoch, opt.max_epoch) if opt.verbose else tqdm(range(opt.start_epoch, opt.max_epoch)):
        if opt.pruning:
            finetune_epoch = 150
            if epoch > finetune_epoch + 1:
                opt.finetune = True
            if epoch == finetune_epoch:
                opt.add_pi = True
            if epoch == finetune_epoch + 1:
                if opt.dptype:
                    model.set_mask_threshold()
                opt.add_pi = False
                opt.use_uniform_mask = True
                opt.mask_type = 'pi_sum'

        model.train() if opt.gpus <= 1 else model.module.train()
        loss_meter.reset()
        accuracy_meter.reset()

        for ii, (input_, target) in enumerate(train_loader):
            print("batch",ii)
            input_, target = input_.to(device), target.to(device)
     
            if opt.add_noisedata:
                noise_mask = np.random.binomial(n=1,p=opt.pr_bernoulli, size=[input_.size(0),1,1,1])
                gaussian_noise = np.random.normal(size=input_.size()) * opt.noise_scalar
                gaussian_noise = torch.from_numpy(gaussian_noise * noise_mask).type_as(input_).to(device)
                input_ = input_ + gaussian_noise
            optimizer.zero_grad()
            model.epoch = epoch



            if opt.GFN_dropout==False:
                
                score = model(input_, target)


                loss = criterion(score, target)
                loss.backward()


                
                ## record gradient
                gradient = torch.zeros([0]).to(device)
                for i in model.parameters():
                    gradient = torch.cat((gradient, i.grad.view(-1)), 0)
                first_order = 0.9999 * first_order + 0.0001 * gradient
                second_order = 0.9999 * second_order + 0.0001 * gradient.pow(2)
                variance = torch.mean(torch.abs(second_order - first_order.pow(2))).item()
                variance_history[total_steps] = variance

                optimizer.step()


                loss_meter.add(loss.cpu().data)
                accuracy_meter.add(score.data, target.data)

            if opt.GFN_dropout==True:
                #update GFN related parameters

                
                metric=model._gfn_step(input_, target,mask_train="",mask=opt.mask)
                    

                loss=metric['CELoss']
                acc=metric['acc']

                loss_meter.add(loss)
                accuracy_meter.add(acc)

                GFNloss_unconditional_meter.add(metric['GFN_loss_unconditional'])
                LogZ_unconditional_meter.add(metric['LogZ_unconditional'])
                LogPF_qz_meter.add(metric['LogPF_qz'])

                GFNloss_conditional_meter.add(metric['GFN_loss_conditional'])
                LogZ_conditional_meter.add(metric['LogZ_conditional'])
                LogPF_qzxy_meter.add(metric['LogPF_qzxy'])

                LogPF_BNN_meter.add(metric['LogPF_BNN'])


                actual_dropout_rate_meter.add(metric['actual_dropout_rate'])

                COR_qz_meter.add(metric['COR_qz'])
                COR_qzxy_meter.add(metric['COR_qzxy'])

                Log_pz_meter.add(metric['Log_pz'])
                Log_pzx_meter.add(metric['Log_pzx'])

            if opt.GFN_dropout==False:
                
                e_fl, e_l0 = model.get_exp_flops_l0() if opt.gpus <= 1 else model.module.get_exp_flops_l0()
     
                # vis.plot('stats_comp/exp_flops', e_fl, total_steps)
                # vis.plot('stats_comp/exp_l0', e_l0, total_steps)

            total_steps += 1
            #determine how many neurons
            #print(ii)
            if opt.GFN_dropout==False:
                if (model.beta_ema if opt.gpus <= 1 else model.module.beta_ema) > 0.:
                    model.update_ema() if opt.gpus <= 1 else model.module.update_ema()

            if ii % opt.print_freq == opt.print_freq - 1:
                # vis.plot('train/loss', loss_meter.value()[0])
                # vis.plot('train/accuracy', accuracy_meter.value()[0])

                if opt.GFN_dropout==True:
                	pass
                    # vis.plot('train/GFNloss_unconditional', GFNloss_unconditional_meter.value()[0])
                    # vis.plot('train/LogZ_unconditional', LogZ_unconditional_meter.value()[0])
                    # vis.plot('train/LogPF_qz', LogPF_qz_meter.value()[0])
                    
                    # vis.plot('train/GFNloss_conditional', GFNloss_conditional_meter.value()[0])
                    # vis.plot('train/LogZ_conditional', LogZ_conditional_meter.value()[0])
                    # vis.plot('train/LogPF_qzxy', LogPF_qzxy_meter.value()[0])

                    # vis.plot('train/LogPF_BNN', LogPF_BNN_meter.value()[0])

                    # vis.plot('train/actual_dropout_rate', actual_dropout_rate_meter.value()[0])


                    # vis.plot('train/COR_qz', COR_qz_meter.value()[0])

                    # vis.plot('train/COR_qzxy', COR_qzxy_meter.value()[0])


                    # vis.plot('train/Log_pz', Log_pz_meter.value()[0])

                    # vis.plot('train/Log_pzx', Log_pzx_meter.value()[0])


                if opt.verbose:
                    if opt.GFN_dropout:
                    	if str(opt.dataset) != "iecu_al":                		
	                        print("epoch:{epoch},lr:{lr},loss:{loss:.2f},train_acc:{train_acc:.2f} GFN_loss_conditional:{GFN_loss_conditional} GFN_loss_unconditional:{GFN_loss_unconditional} actual_dropout_rate:{actual_dropout_rate}"
	                                                  .format(epoch=epoch, loss=loss_meter.value()[0],
	                                                          train_acc=accuracy_meter.value()[0],
	                                                          lr=optimizer.param_groups[0]['lr'],
	                                                          GFN_loss_conditional=metric['GFN_loss_conditional'],
	                                                          GFN_loss_unconditional=metric['GFN_loss_unconditional'],
	                                                          actual_dropout_rate=metric['actual_dropout_rate'],))

                    else:
                    	if str(opt.dataset) != "iecu_al":                		
	                        print("epoch:{epoch},lr:{lr},loss:{loss:.2f},train_acc:{train_acc:.2f}"
	                          .format(epoch=epoch, loss=loss_meter.value()[0],
	                                  train_acc=accuracy_meter.value()[0],
	                                  lr=optimizer.param_groups[0]['lr']))

        # save model
       
        if epoch % 10 == 0 or epoch == opt.max_epoch - 1:
            torch.save(model.state_dict(), directory + '/{}.model'.format(epoch))
            torch.save(optimizer.state_dict(), directory + '/{}.optim'.format(epoch))

        # validate model
        val_accuracy, val_loss, label_dict, input__dict, logits_dict, logits_dict_greedy, base_aic, up, ucpred, ac_prob, iu_prob, elbo, ece, predicted_labels_dict = val(model, val_loader, criterion, num_classes, opt)


        if val_loss<best_val_loss:
            torch.save(model.state_dict(), directory + '/best.model')
            torch.save(optimizer.state_dict(), directory + '/best.optim')
            best_val_loss=val_loss

        # if opt.recorddp:
        #     if epoch == 10:
        #         dp_history[total_steps] = dprate
        #         target_history[total_steps] = target.cpu().numpy()
        #         #input__history[total_steps] = input_.cpu().numpy()
        val_accuracy_history[total_steps] = {'accuracy': val_accuracy, 'aic': base_aic, 'up': up, 'ucpred': ucpred, 'ac_prob': ac_prob, 'iu_prob': iu_prob, 'elbo': elbo, 'ece': ece}


        # vis.plot('val/loss', val_loss)
        # vis.plot('val/accuracy', val_accuracy)

        # update lr
        if scheduler is not None:
            if isinstance(optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if opt.GFN_dropout==True:
            model.taskmodel_scheduler.step()

            ####gradually improve beta (= decrease temperature) to allow the GFN to find different modes
            if epoch!=0 and epoch%30==0:
                model.beta=min(3.16*model.beta,1.0) #1000^(1/6)~3.16, assumging there are 200 epochs



        if opt.verbose:
            if opt.GFN_dropout==False:
            	if opt.is_iecu:

             		if str(opt.dataset) != "iecu_al":
	                	print("epoch:{epoch},lr:{lr},loss:{loss:.2f}, recall:{val_acc:.2f}, uncer:{base_aic_1:.2f}, {base_aic_2:.2f},{base_aic_3:.2f}, "
	                      "up:{up_1:.2f}, {up_2:.2f},{up_3:.2f}, ucpred:{ucpred_1:.2f}, {ucpred_2:.2f},{ucpred_3:.2f}, "
	                      "ac_prob:{ac_prob_1:.2f}, {ac_prob_2:.2f},{ac_prob_3:.2f}, iu_prob:{iu_prob_1:.2f}, {iu_prob_2:.2f},{iu_prob_3:.2f}, elbo:{elbo:.2f}, prune_rate:{pr:.2f}, ece:{ece:.4f}"
	                      .format(epoch=epoch, loss=loss_meter.value()[0], val_acc=val_accuracy[1], base_aic_1=base_aic[0], base_aic_2=base_aic[1],
	                              base_aic_3=base_aic[2], up_1=up[0],up_2=up[1],up_3=up[2],ucpred_1 = ucpred[0], ucpred_2 = ucpred[1], ucpred_3 = ucpred[2],
	                              ac_prob_1=ac_prob[0],ac_prob_2=ac_prob[1],ac_prob_3=ac_prob[2],iu_prob_1=iu_prob[0],iu_prob_2=iu_prob[1],iu_prob_3=iu_prob[2],elbo=elbo,
	                              lr=optimizer.param_groups[0]['lr'], pr=model.prune_rate() if opt.gpus <= 1 else model.module.prune_rate(), ece = ece[0]))
            else:
            	if str(opt.dataset) != "iecu_al":        		
	                print("epoch:{epoch},lr:{lr},loss:{loss:.2f},recall:{val_acc:.2f}, uncer:{base_aic_1:.2f}, {base_aic_2:.2f},{base_aic_3:.2f}, "
	                      "up:{up_1:.2f}, {up_2:.2f},{up_3:.2f}, ucpred:{ucpred_1:.2f}, {ucpred_2:.2f},{ucpred_3:.2f}, "
	                      "ac_prob:{ac_prob_1:.2f}, {ac_prob_2:.2f},{ac_prob_3:.2f}, iu_prob:{iu_prob_1:.2f}, {iu_prob_2:.2f},{iu_prob_3:.2f}, elbo:{elbo:.2f}, ece:{ece:.4f}"
	                      .format(epoch=epoch, loss=loss_meter.value()[0], val_acc=val_accuracy[1], base_aic_1=base_aic[0], base_aic_2=base_aic[1],
	                              base_aic_3=base_aic[2], up_1=up[0],up_2=up[1],up_3=up[2],ucpred_1 = ucpred[0], ucpred_2 = ucpred[1], ucpred_3 = ucpred[2],
	                              ac_prob_1=ac_prob[0],ac_prob_2=ac_prob[1],ac_prob_3=ac_prob[2],iu_prob_1=iu_prob[0],iu_prob_2=iu_prob[1],iu_prob_3=iu_prob[2],elbo=elbo,
	                              lr=model.taskmodel_optimizer.param_groups[0]['lr'], ece = ece[0]))
                   
        #for (i, num) in enumerate(model.get_expected_activated_neurons() if opt.gpus <= 1
        #                          else model.module.get_expected_activated_neurons()):
        #    vis.plot("Training_layer/{}".format(i), num)
        # vis.plot('lr', optimizer.param_groups[0]['lr'])


       # histories['dp_history'] = dp_history
        histories['target_history'] = target_history
        histories['input__history'] = input__history
        histories['val_accuracy_history'] = val_accuracy_history
        histories['first_order_history'] = first_order.data.cpu().numpy()
        histories['second_order_history'] = second_order.data.cpu().numpy()
        histories['variance_history'] = variance_history
        # histories['variance'] = 0
        #print('var', variance)
        with open(os.path.join(directory, 'histories_' + '.pkl'), 'wb') as f:
            cPickle.dump(histories, f)

    return directory

def two_sample_test_batch(logits):
    prob = torch.softmax(logits, 1)
    probmean = torch.mean(prob,2)
    values, indices = torch.topk(probmean, 2, dim=1)
    aa = logits.gather(1, indices[:,0].unsqueeze(1).unsqueeze(1).repeat(1,1,opt.sample_num))
    bb = logits.gather(1, indices[:,1].unsqueeze(1).unsqueeze(1).repeat(1,1,opt.sample_num))
    if opt.t_test:
        pvalue = sts.ttest_rel(aa, bb, axis=2).pvalue
    else:
        pvalue = np.zeros(shape=(aa.shape[0], aa.shape[1]))
        for i in range(pvalue.shape[0]):
            pvalue = sts.wilcoxon(aa[i, 0, :], bb[i, 0, :]).pvalue
    return pvalue

def val(model, dataloader, criterion, num_classes, opt):
    # also return the label (batch size), and k sampled logits (batch_size, num_classes, k)
    model.eval() if opt.gpus <= 1 else model.module.eval()
    loss_meter = meter.AverageValueMeter()
    loss_meter_greedy = meter.AverageValueMeter()
    accuracy_meter = meter.ClassErrorMeter(accuracy=True)
    accuracy_meter_greedy = meter.ClassErrorMeter(accuracy=True)
    logits_dict = OrderedDict()
    label_dict = OrderedDict()
    input__dict = OrderedDict()
    predicted_labels_dict = OrderedDict()
    logits_dict_greedy = OrderedDict()
    accurate_pred = torch.zeros([0], dtype=torch.float64)
    testresult = torch.zeros([0], dtype=torch.float64)
    noise_mask_conca = torch.zeros([0], dtype=torch.float64)
    elbo_list = []
    label_tensors = torch.zeros([0], dtype=torch.int64)
    score_tensors = torch.zeros([0], dtype=torch.float32)

    f1_preds, recall_preds, precision_preds, balanced_accs = [], [], [], []
    
    for ii, data in enumerate(dataloader):
        input_, label = data
        input_, label = input_.to(device), label.to(device)

        if opt.augment_test:
            
            augmented_images=torch.zeros([0], dtype = input_.dtype)
            for j in range(input_.shape[0]):
                #angle=randrange(20)#random rotation range
                #angle=0.5 #random rotation range

                

                #rotated_tensor= image_rotate(input_[j,:,:,:].cpu(), angle*360.0/20.0)
                augmented_tensor= image_augmentation(input_[j,:,:,:].cpu())
                
                # if j==1:
                #     from torchvision.utils import save_image
                #     img_before = input_[j,:,:,:]
                #     img_after=augmented_tensor
                #     save_image(img_before, 'img_before.png')
                #     save_image(img_after, 'img_after.png')
                    
                #output_tensor_i = image_rotate(inputs, i * 360.0 / 20.0)
                augmented_images = torch.cat([augmented_images, augmented_tensor.unsqueeze(0)],0)

            input_ = augmented_images.to(device)
        else:
            pass

        logits_ii = np.zeros([input_.size(0), num_classes, opt.sample_num])
        logits_greedy = np.zeros([input_.size(0), num_classes])

        # greedy
        opt.test_sample_mode = 'greedy'
        opt.use_t_in_testing = True
        noise_mask = np.zeros(shape=[input_.size(0), 1, 1, 1])

            
        if opt.add_noisedata:
            noise_mask = np.random.binomial(n=1, p=opt.pr_bernoulli, size=[input_.size(0), 1, 1, 1])
            gaussian_noise = np.random.normal(size=input_.size()) * opt.noise_scalar
            gaussian_noise = torch.from_numpy(gaussian_noise * noise_mask).type_as(input_).to(device)
            input_ = input_ + gaussian_noise
        # input_ = input_ + torch.from_numpy(np.random.normal(size=input_.size())).to(device)*opt.noise
        if opt.GFN_dropout==False:
            score = model(input_, label)
        if opt.GFN_dropout==True:
            score,actual_masks,masks_qz,masks_qzxy,LogZ_unconditional,LogPF_qz,LogR_qz,LogPB_qz,LogPF_BNN,LogZ_conditional,LogPF_qzxy,LogR_qzxy,LogPB_qzxy,Log_pzx,Log_pz  = model.GFN_forward(input_,label,mask=opt.mask)
        

        ####
        label_tensors = torch.cat((label_tensors, label.cpu()), 0)
        score_tensors = torch.cat((score_tensors, score.detach().cpu()), 0)
        ####
        logits_greedy[:, :] = score.cpu().data.numpy()
        logits_dict_greedy[ii] = logits_greedy
        mean_logits_greedy=torch.from_numpy(logits_greedy).to(device)
        accuracy_meter_greedy.add(mean_logits_greedy.squeeze(), label.long())
        loss_greedy = criterion(mean_logits_greedy, label)
        loss_meter_greedy.add(loss_greedy.cpu().data)
        #sample
        opt.test_sample_mode = 'sample'
        opt.use_t_in_testing = False
        for iii in range(opt.sample_num):
            # important step !!!!!!
            if opt.GFN_dropout==True:
                score = model(input_, label,opt.mask)
            else:
                score = model(input_, label)
            
            logits_ii[:, :, iii] = score.cpu().data.numpy()
            elbo_list.append(model.elbo.cpu().numpy())
        logits_dict[ii] = logits_ii
        label_dict[ii] = label.cpu()
        input__dict[ii] = input_.cpu().numpy()
    #TODO: should I average logits or probabilities
        mean_logits = F.log_softmax(torch.mean(F.softmax(torch.from_numpy(logits_ii).to(device), dim=1), 2), 1)
        accuracy_meter.add(mean_logits.squeeze(), label.long())
        loss = criterion(mean_logits, label)
        loss_meter.add(loss.cpu().data)
        logits_tsam = torch.from_numpy(logits_ii)
        prob = F.softmax(logits_tsam, 1)
        ave_prob = torch.mean(prob, 2)
        # prediction = torch.argmax(ave_prob, 1).to(device)
        prediction = torch.argmax(torch.from_numpy(logits_greedy), 1).to(device) #TODO: use greedy or sample?
        
        # F1-score
        pred_idx = prediction.detach().cpu().numpy()
        labels = label.detach().cpu().numpy()
        predicted_labels_dict[ii] = pred_idx
        f1_preds.append(f1_score(labels, pred_idx, average='macro'))
        recall_preds.append(recall_score(labels, pred_idx, average='macro'))
        precision_preds.append(precision_score(labels, pred_idx, average='macro'))
        balanced_accs.append(balanced_accuracy_score(labels, pred_idx))

        #################################################################

        accurate_pred_i = (prediction == label).type_as(logits_tsam)
        accurate_pred = torch.cat([accurate_pred, accurate_pred_i], 0)
        testresult_i = torch.from_numpy(two_sample_test_batch(logits_tsam)).type_as(logits_tsam)
        testresult = torch.cat([testresult, testresult_i], 0)
        noise_mask_conca = torch.cat([noise_mask_conca, torch.from_numpy(noise_mask[:,0,0,0]).type_as(logits_tsam)], 0)

    ####
    ece = ECELoss(n_bins = 10)(score_tensors, label_tensors)
    ####
    uncertain = (testresult > 0.01).type_as(mean_logits).cpu()
    up_1 = uncertain.mean() * 100
    ucpred_1 = ((uncertain == noise_mask_conca).type_as(mean_logits)).mean() * 100
    ac_1 = (accurate_pred * (1 - uncertain.squeeze())).sum()
    iu_1 = ((1 - accurate_pred) * uncertain.squeeze()).sum()

    ac_prob_1 = ac_1 / (1 - uncertain.squeeze()).sum() * 100
    iu_prob_1 = iu_1 / (1 - accurate_pred).sum() * 100



    uncertain = (testresult > 0.05).type_as(mean_logits).cpu()
    up_2 = uncertain.mean() * 100
    ucpred_2 = (uncertain == noise_mask_conca).type_as(mean_logits).mean() * 100
    ac_2 = (accurate_pred * (1 - uncertain.squeeze())).sum()
    iu_2 = ((1 - accurate_pred) * uncertain.squeeze()).sum()

    ac_prob_2 = ac_2 / (1 - uncertain.squeeze()).sum() * 100
    iu_prob_2 = iu_2 / (1 - accurate_pred).sum() * 100

    uncertain = (testresult > 0.1).type_as(mean_logits).cpu()
    up_3 = uncertain.mean() * 100
    ucpred_3 = (uncertain == noise_mask_conca).type_as(mean_logits).mean() * 100
    ac_3 = (accurate_pred * (1 - uncertain.squeeze())).sum()
    iu_3 = ((1 - accurate_pred) * uncertain.squeeze()).sum()

    ac_prob_3 = ac_3 / (1 - uncertain.squeeze()).sum() * 100
    iu_prob_3 = iu_3 / (1 - accurate_pred).sum() * 100

    base_aic_1 = (ac_1 + iu_1) / accurate_pred.size(0) * 100
    base_aic_2 = (ac_2 + iu_2) / accurate_pred.size(0) * 100
    base_aic_3 = (ac_3 + iu_3) / accurate_pred.size(0) * 100
    base_aic = [base_aic_1,base_aic_2,base_aic_3]

    ac_prob = [ac_prob_1, ac_prob_2, ac_prob_3]
    iu_prob = [iu_prob_1, iu_prob_2, iu_prob_3]
    ucpred = [ucpred_1, ucpred_2, ucpred_3]

    # uncertainty proportion
    up = [up_1,up_2,up_3]

    #for (i, num) in enumerate(model.get_activated_neurons() if opt.gpus <= 1 else model.module.get_activated_neurons()):
    #    vis.plot("val_layer/{}".format(i), num)

    #for (i, z_phi) in enumerate(model.z_phis()):
    #    if opt.hardsigmoid:
    #        vis.hist("hard_sigmoid(phi)/{}".format(i), F.hardtanh(opt.k * z_phi / 7. + .5, 0, 1).cpu().detach().numpy())
    #    else:
    #        vis.hist("sigmoid(phi)/{}".format(i), torch.sigmoid(opt.k * z_phi).cpu().detach().numpy())
    if opt.GFN_dropout==False:
    	pass
        # vis.plot("prune_rate", model.prune_rate() if opt.gpus <= 1 else model.module.prune_rate())
    #return accuracy_meter.value()[0], loss_meter.value()[0], label_dict, logits_dict
    if opt.is_iecu:
        return (np.mean(f1_preds), np.mean(recall_preds), np.mean(precision_preds), np.mean(balanced_accs)), loss_meter_greedy.value()[0], label_dict, input__dict, logits_dict, logits_dict_greedy, base_aic, up, ucpred, ac_prob, iu_prob, np.mean(elbo_list)*100, ece, predicted_labels_dict
    else:
        return accuracy_meter_greedy.value()[0], loss_meter_greedy.value()[0], label_dict, input__dict, logits_dict, logits_dict_greedy, base_aic, up, ucpred, ac_prob, iu_prob, np.mean(elbo_list)*100, ece, predicted_labels_dict
    #accuracy_meter.value()[0], loss_meter.value()[0]


def test(**kwargs):
    opt.parse(kwargs)
    global device
    #device = torch.device("cuda" if opt.use_gpu else "cpu")
    # vis = Visualizer(opt.log_dir, opt.model, current_time)
    # # load model
    # model = getattr(models, opt.model)(lambas=opt.lambas).to(device)
    # # load data set
    # train_loader, test_loader, num_classes = getattr(dataset, opt.dataset)(opt.batch_size)
    train_loader,val_loader, test_loader, num_classes = getattr(dataset, opt.dataset)(opt.batch_size)
    # load model
    print("***********************opt.model****")
    print(opt.model)
    model = getattr(models, opt.model)(lambas=opt.lambas, num_classes=num_classes, weight_decay=opt.weight_decay,opt=opt).to(
        device)

    #directory = dir + '{}/{}'.format(opt.checkpoints_dir, opt.model)
    if opt.valtestseed is not None:
        setup_seed_2(opt.valtestseed)
    # define loss function
    def criterion(output, target_var):
        loss = nn.CrossEntropyLoss().to(device)(output, target_var.long())
        if opt.GFN_dropout:
            total_loss =loss
        else:
            total_loss = (loss + model.regularization() if opt.gpus <= 1 else model.module.regularization()).to(device)
        return total_loss

    if len(opt.load_file) > 0:
        model.load_state_dict(torch.load(opt.load_file,map_location=device))
        val_accuracy, val_loss, label_dict, input__dict, logits_dict, logits_dict_greedy, base_aic, up, ucpred, ac_prob, iu_prob, elbo, ece, predicted_labels_dict = val(model, test_loader, criterion, num_classes, opt)
        
        if opt.GFN_dropout==False:
            if opt.is_iecu:
                print('Macro F1-score: {}, Macro Recall: {}, Mean Precision: {}, Mean Balanced Accuracy: {}'.format(val_accuracy[0], val_accuracy[1], val_accuracy[2], val_accuracy[3]))
            else:            
                print("augment_test:{augment_test} loss:{loss:.2f},val_acc:{val_acc:.2f}, uncer:{base_aic_1:.2f}, {base_aic_2:.2f},{base_aic_3:.2f}, up:{up_1:.2f}, {up_2:.2f},{up_3:.2f}, prune_rate:{pr:.2f}, elbo:{elbo:.2f}, ece:{ece:.4f}"
                      .format(augment_test=opt.augment_test,loss=val_loss, val_acc=val_accuracy, base_aic_1=base_aic[0], base_aic_2=base_aic[1],
                                  base_aic_3=base_aic[2], up_1=up[0],up_2=up[1],up_3=up[2], elbo=elbo , pr=model.prune_rate()if opt.gpus <= 1 else model.module.prune_rate(), ece = ece[0]))
        else:
            if opt.is_iecu:
                print('Macro F1-score: {}, Macro Recall: {}, Mean Precision: {}, Mean Balanced Accuracy: {}'.format(val_accuracy[0], val_accuracy[1], val_accuracy[2], val_accuracy[3]))
            else:
                print("augment_test:{augment_test} loss:{loss:.2f},val_acc:{val_acc:.2f}, uncer:{base_aic_1:.2f}, {base_aic_2:.2f},{base_aic_3:.2f}, up:{up_1:.2f}, {up_2:.2f},{up_3:.2f}, elbo:{elbo:.2f}, ece:{ece:.4f}"
                      .format(augment_test=opt.augment_test,loss=val_loss, val_acc=val_accuracy, base_aic_1=base_aic[0], base_aic_2=base_aic[1],
                                  base_aic_3=base_aic[2], up_1=up[0],up_2=up[1],up_3=up[2], elbo=elbo , ece = ece[0]))


        # print(model.get_activated_neurons())
    test_result = {'label': label_dict, 'logits': [logits_dict, logits_dict_greedy], 'input': input__dict}
    with open(os.path.join(opt.load_file + 'test_result' + '.pkl'), 'wb') as f:
        cPickle.dump(test_result, f)

    #save results as csv
    import csv
    results_tosave=[opt.load_file,opt.mask,opt.BNN,opt.augment_test,val_loss.item(),val_accuracy, base_aic[0].item(),
                     base_aic[1].item(),
                    base_aic[2].item(), up[0].item(),
                    up[1].item(),up[2].item(), elbo, ece[0].item()]    
    with open(r'testresult.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(results_tosave)


def image_rotate(inputs,angle):
    inputs = inputs.squeeze()
    im = transforms.ToPILImage()(inputs)
    grouped_z1_pil = torchvision.transforms.functional.rotate(im, float(angle))
    output_tensor = torchvision.transforms.ToTensor()(grouped_z1_pil)
    return output_tensor


def val_rotate(model, dataloader, num_classes, opt):
    # also return the label (batch size), and k sampled logits (batch_size, num_classes, k)
    model.eval() if opt.gpus <= 1 else model.module.eval()
    loss_meter = meter.AverageValueMeter()
    accuracy_meter = meter.ClassErrorMeter(accuracy=True)
    logits_dict = OrderedDict()
    label_dict = OrderedDict()

    for ii, data in enumerate(dataloader):
        input_, label = data
        input_, label = input_.to(device), label.to(device)
        if ii >0:
            break
    input_[3,:,:,:]
    inputs =input_[3,:,:,:].unsqueeze(0)


    rotate_image = torch.zeros([0], dtype = inputs.dtype)
    for i in range(20):
        output_tensor_i = image_rotate(inputs.cpu(), i*360.0/20.0)
        #output_tensor_i = image_rotate(inputs, i * 360.0 / 20.0)
        rotate_image = torch.cat([rotate_image, output_tensor_i.unsqueeze(0)],0)
    rotate_image = rotate_image.cuda()

    logits_ii = np.zeros([rotate_image.size(0), num_classes, opt.sample_num])
    # sample
    for iii in range(opt.sample_num):
        # important step !!!!!!
        score = model(rotate_image, label)
        logits_ii[:, :, iii] = score.cpu().data.numpy()
    logits_dict[0] = logits_ii
    label_dict[0] = label.cpu()
#TODO: should I average logits or probabilities
    return rotate_image, label_dict, logits_dict


def test_rotate(**kwargs):
    opt.parse(kwargs)
    global device
    #device = torch.device("cuda" if opt.use_gpu else "cpu")
    # vis = Visualizer(opt.log_dir, opt.model, current_time)
    # load model
    model = getattr(models, opt.model)(lambas=opt.lambas,opt=opt).to(device)
    # load data set
    train_loader, test_loader, num_classes = getattr(dataset, opt.dataset)(opt.batch_size)
    #directory = dir + '{}/{}'.format(opt.checkpoints_dir, opt.model)
    # define loss function
    def criterion(output, target_var):
        loss = nn.CrossEntropyLoss().to(device)(output, target_var.long())
        total_loss = (loss + model.regularization() if opt.gpus <= 1 else model.module.regularization()).to(device)
        return total_loss

    if len(opt.load_file) > 0:
        model.load_state_dict(torch.load(opt.load_file,map_location=device))
        rotate_image, label_dict, logits_dict = val_rotate(model, test_loader, num_classes, opt)
        # print("loss:{loss:.2f},val_acc:{val_acc:.2f},prune_rate:{pr:.2f}"
        #       .format(loss=val_loss, val_acc=val_accuracy,
        #               pr=model.prune_rate() if opt.gpus <= 1 else model.module.prune_rate()))
        # print(model.get_activated_neurons())
    test_result = {'label': label_dict, 'logits': logits_dict}
    test_result = {'rotate': rotate_image, 'label': label_dict, 'logits': logits_dict}
    print(test_result)
    with open(os.path.join(opt.load_file + 'test_result' + '.pkl'), 'wb') as f:
        cPickle.dump(test_result, f)

def dempster_shafer(logits):
	logits = torch.tensor(logits.mean(-1))
	num_classes = logits.shape[1]
	belief_mass = logits.exp().sum(1)
	return num_classes / (belief_mass + num_classes)

def uncertainty_logits(input_dict, dict_logits, top_k):
	uncertainty_dict = []
	for index, ((_, input_), (_, logit)) in enumerate(zip(input_dict.items(), dict_logits.items())):
		uncertainty_dict.append((input_, dempster_shafer(logit).item(), index))
	return sorted(uncertainty_dict, key=lambda item: item[1], reverse=True)[:top_k]

def th_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

def augment_new_dataset(directory, **kwargs):
    global device
    opt.parse(kwargs)

    # evaluate the model on the augmenting set    
    def criterion(output, target_var):
        loss = nn.CrossEntropyLoss().to(device)(output, target_var.long())
        if opt.GFN_dropout:
            total_loss = loss
        else:
            total_loss = (loss + model.regularization() if opt.gpus <= 1 else model.module.regularization()).to(device)
        return total_loss

    # vis = Visualizer(opt.log_dir, opt.model, current_time)
    train_loader, val_loader, test_loader, num_classes = getattr(dataset, opt.dataset)(opt.batch_size)
    model = getattr(models, opt.model)(lambas=opt.lambas, num_classes=num_classes, weight_decay=opt.weight_decay,
                                       opt=opt).to(
        device)
    model.load_state_dict(torch.load(directory, map_location=device))
    _, _, label_dict, input__dict, logits_dict, _, _, _, _, _, _, _, _, predicted_labels_dict = val(model, test_loader,
                                                                                                    criterion,
                                                                                                    num_classes,
                                                                                                    opt)
    uncertainties = uncertainty_logits(input__dict, logits_dict, 2000)
    # append the new high-uncertain inputs to the the training dataset
    X = train_loader.dataset.getX()
    Y = train_loader.dataset.getY()
    # indices of elements to remove from augmenting set
    indices_to_delete = []
    index_count = 0
    for ((input_, uncertainty, index), key_label) in zip(uncertainties, predicted_labels_dict.items()):
        if index_count > 1999:  # index starts by 0 so 0-1999 = 2k elements 
            break
        else:
            label_ = list(key_label)[1]
            indices_to_delete.append(index)
            index_count += 1
        X = torch.cat((X, torch.tensor(input_).float()), 0)
        Y = torch.cat((Y, torch.tensor(label_).float()), 0)

    # Remove the top-k from the augmenting dataset
    X_aug = test_loader.dataset.getX()
    Y_aug = test_loader.dataset.getY()
    X_aug = th_delete(X_aug, indices_to_delete)
    Y_aug = th_delete(Y_aug, indices_to_delete)
    test_loader.dataset.setX(X_aug)
    test_loader.dataset.setY(Y_aug)

    train_loader.dataset.setX(X)
    train_loader.dataset.setY(Y)
    
    print('New size of dataset: {}'.format(len(train_loader)))
    kwargs['new_data'] = (train_loader, val_loader, test_loader, num_classes)
    return kwargs

def get_al_test_loader():
    path_test_set = "/home/mila/b/bonaventure.dossou/iecu_al_test.csv"
    testing_set = pd.read_csv(path_test_set, delimiter=",")
    testing_targets = testing_set["Death"].values

    testing_set.drop(columns=['Death'], inplace=True)
    testing_features = testing_set.values[:, :1369]

    if opt.model == "MLP_GFN":
        testing_features = testing_features.reshape(-1, 37, 37)

    iecu_test_dataset = dataset.Dataset(torch.tensor(testing_features).float(), torch.tensor(testing_targets))
    test_loader = torch.utils.data.DataLoader(iecu_test_dataset, batch_size=len(iecu_test_dataset), shuffle=False)
    return test_loader

def active_learning(**kwargs):
    global device
    opt.parse(kwargs)

    # testing AL set
    real_test_loader = get_al_test_loader()

    def criterion(output, target_var):
        loss = nn.CrossEntropyLoss().to(device)(output, target_var.long())
        if opt.GFN_dropout:
            total_loss = loss
        else:
            total_loss = (loss + model.regularization() if opt.gpus <= 1 else model.module.regularization()).to(device)
        return total_loss

    # finetune on the starting set
    kwargs['al_round'] = 0
    directory_model = train(**kwargs)
    directory_model = directory_model + '/best.model'
    dict_of_scores = {}
    for al_round in range(opt.al_rounds):

        # test the model on the test dataset
        model = getattr(models, opt.model)(lambas=opt.lambas, num_classes=2, weight_decay=opt.weight_decay,
                                           opt=opt).to(device)
        model.load_state_dict(torch.load(directory_model, map_location=device))
        metrics, _, _, _, logits_dict, _, _, _, _, _, _, _, _, _ = val(model, real_test_loader, criterion, 2,
                                                                       opt)
        logits_ = list(logits_dict.items())
        dict_of_scores['round_{}'.format(al_round)] = (
        metrics[0], metrics[1], metrics[2], metrics[3], dempster_shafer(logits_[0][1]).mean().item())

        # augment data
        kwargs = augment_new_dataset(directory_model, **kwargs)
        kwargs['al_round'] = al_round + 1
        opt.parse(kwargs)

        # train on new dataset
        print('Active Learning Round: {}'.format(al_round + 1))
        directory_model = train(active_learning=True, **kwargs)
        directory_model = directory_model + '/best.model'


    # testing the AL model of the last round
    # directory_model = directory_model + '/best.model'
    opt.parse(kwargs)

    # train_loader, val_loader, test_loader, num_classes = getattr(dataset, opt.dataset)(opt.batch_size)
    model = getattr(models, opt.model)(lambas=opt.lambas, num_classes=2, weight_decay=opt.weight_decay,
                                       opt=opt).to(device)
    model.load_state_dict(torch.load(directory_model, map_location=device))
    metrics, _, _, _, logits_dict, _, _, _, _, _, _, _, _, _ = val(model, real_test_loader, criterion, 2, opt)
    logits_ = list(logits_dict.items())
    dict_of_scores['round_{}'.format(opt.al_rounds)] = (
    metrics[0], metrics[1], metrics[2], metrics[3], dempster_shafer(logits_[0][1]).mean().item())

    # frame = pd.DataFrame()
    # frame['method'] = [str(opt.model_name)]
    # frame['al_f1_score'] = [metrics[0]]
    # frame['al_recall_score'] = [metrics[1]]
    # frame['al_precision_score'] = [metrics[2]]
    # frame['al_balAcc_score'] = [metrics[3]]
    # logits_ = list(logits_dict.items())
    # frame['dempster_shafer_uncertainty'] = [dempster_shafer(logits_[0][1]).mean().item()]
    if opt.model_name != "MLP_GFN":
        with open('/home/mila/b/bonaventure.dossou/GFNDropout/tasks/Scripts/al_{}.json'.format(str(opt.model_name)),
                  'w') as f:
            json.dump(dict_of_scores, f, indent=4)
        f.close()
    # frame.to_csv('/home/mila/b/bonaventure.dossou/GFNDropout/tasks/Scripts/{}.csv'.format(),
    #             index=False)
    else:
        with open('/home/mila/b/bonaventure.dossou/GFNDropout/tasks/Scripts/al_{}_{}.json'.format(str(opt.model_name),
                                                                                                  str(opt.mask)),
                  'w') as f:
            json.dump(dict_of_scores, f, indent=4)
        f.close()
    # frame.to_csv('/home/mila/b/bonaventure.dossou/GFNDropout/tasks/Scripts/{}_{}.csv'.format(str(opt.model_name), str(opt.mask)),index=False)

def help():
    '''help'''
    print('''
    usage : python main.py <function> [--args=value]
    <function> := train | test | active_learning | help
    example: 
            python {0} train --model=ARMLeNet5 --dataset=mnist --lambas='[.1,.1,.1,.1]' --optimizer=adam --lr=0.001
            python {0} test --model=ARMLeNet5 --dataset=mnist --lambas='[.1,.1,.1,.1]' --load_file="checkpoints/ARMLeNet5_2019-06-19 14:27:03/0.model"
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire
    fire.Fire({'train': train, 'test': test, 'active_learning': active_learning, 'help': help, 'test_rotate': test_rotate})
