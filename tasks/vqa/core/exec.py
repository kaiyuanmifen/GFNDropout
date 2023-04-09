# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.data.load_data import DataSet
from core.model.net import Net
from core.model.optim import get_optim, adjust_lr
from core.data.data_utils import shuffle_list
from utils.vqa import VQA
from utils.vqaEval import VQAEval

import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import scipy.stats as sts
from six.moves import cPickle
from ece import ECELoss

import torch.optim as optimizer

def setup_seed_2(seed):
    np.random.seed(seed)


class Execution:
    def __init__(self, HP):
        self.HP = HP

        print('Loading training set ........')
        self.dataset = DataSet(HP)

        self.dataset_eval = None
        if HP.EVAL_EVERY_EPOCH:
            HP_eval = copy.deepcopy(HP)
            setattr(HP_eval, 'RUN_MODE', 'val')

            print('Loading validation set for per-epoch evaluation ........')
            self.dataset_eval = DataSet(HP_eval)


    def train(self, dataset, dataset_eval=None):
        # Obtain needed information
        setup_seed_2(self.HP.SEED)



        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb
        self.HP.data_size = data_size
        print(data_size)
        # Define the MCAN model
        net = Net(
            self.HP,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.train()
        if self.HP.ARM and self.HP.dp_type:
            self.dropout_list = []
            self.dropout_list = self.dropout_list + [enc.mhatt.dropout for enc in net.backbone.enc_list]
            self.dropout_list = self.dropout_list +([enc.ffn.mlp.fc.dropout for enc in net.backbone.enc_list])
            self.dropout_list = self.dropout_list +([enc.dropout1 for enc in net.backbone.enc_list])
            self.dropout_list = self.dropout_list +([enc.dropout2 for enc in net.backbone.enc_list])
            self.dropout_list = self.dropout_list +([dec.mhatt1.dropout for dec in net.backbone.dec_list])
            self.dropout_list = self.dropout_list +([dec.mhatt2.dropout for dec in net.backbone.dec_list])
            self.dropout_list = self.dropout_list +([dec.ffn.mlp.fc.dropout for dec in net.backbone.dec_list])
            self.dropout_list = self.dropout_list +([dec.dropout1 for dec in net.backbone.dec_list])
            self.dropout_list = self.dropout_list +([dec.dropout2 for dec in net.backbone.dec_list])
            self.dropout_list = self.dropout_list +([dec.dropout3 for dec in net.backbone.dec_list])
            self.dropout_list = self.dropout_list +([net.attflat_img.mlp.fc.dropout, net.attflat_lang.mlp.fc.dropout])

        # Define the multi-gpu training if needed
        if self.HP.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.HP.DEVICES)
        self.net = net
        # Define the binary cross entropy loss
        # loss_fn = torch.nn.BCELoss(size_average=False).cuda()
        loss_fn = torch.nn.BCELoss(reduction='sum').cuda()
        loss_fn_keep = torch.nn.BCELoss(reduction='none').cuda()

        pytorch_total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('************total params size: ', pytorch_total_params)
        # Load checkpoint if resume training
        if self.HP.RESUME:
            print(' ========== Resume training')

            if self.HP.CKPT_PATH is not None:
                print('Warning: you are now using CKPT_PATH args, '
                      'CKPT_VERSION and CKPT_EPOCH will not work')

                path = self.HP.CKPT_PATH
            else:
                path = self.HP.CKPTS_PATH + \
                       'ckpt_' + self.HP.CKPT_VERSION +"_seed_"+str(self.HP.SEED)+ \
                       '/epoch' + str(self.HP.CKPT_EPOCH) + '.pkl'

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')
            net.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.HP, net, data_size, ckpt['lr_base'])
            optim._step = int(data_size / self.HP.BATCH_SIZE * self.HP.CKPT_EPOCH)
            optim.optimizer.load_state_dict(ckpt['optimizer'])

            start_epoch = self.HP.CKPT_EPOCH

        else:
            if ('ckpt_' + self.HP.VERSION+"_seed_"+str(self.HP.SEED)) in os.listdir(self.HP.CKPTS_PATH):
                shutil.rmtree(self.HP.CKPTS_PATH + 'ckpt_' + self.HP.VERSION+"_seed_"+str(self.HP.SEED))

            os.mkdir(self.HP.CKPTS_PATH + 'ckpt_' + self.HP.VERSION+"_seed_"+str(self.HP.SEED))

            optim = get_optim(self.HP, net, data_size)
            start_epoch = 0


        if self.HP.GFlowOut!="none":
            ####GFN related optimizer
            z_lr=1e-1
            mg_lr_z=1e-3
            mg_lr_mu=1e-3
            
            q_z_param_list = [  {'params': net.backbone.dec_list_GFN[0].q_z_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},
                                {'params': net.backbone.dec_list_GFN[1].q_z_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},
                                {'params': net.backbone.dec_list_GFN[2].q_z_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},
                                {'params': net.backbone.LogZ_unconditional, 'lr': z_lr,"weight_decay":0.1}]
            q_z_optimizer = optimizer.Adam(q_z_param_list)


            p_zx_param_list = [{'params': net.backbone.dec_list_GFN[0].p_zx_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},
                                {'params': net.backbone.dec_list_GFN[1].p_zx_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},
                                {'params': net.backbone.dec_list_GFN[2].p_zx_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},]
            p_zx_optimizer = optimizer.Adam(p_zx_param_list)
            
            
            q_zxy_param_list = [{'params': net.backbone.dec_list_GFN[0].q_zxy_mask_generators.parameters(), 'lr': mg_lr_z,"weight_decay":0.1},
                                {'params': net.backbone.dec_list_GFN[1].q_zxy_mask_generators.parameters(), 'lr': mg_lr_z,"weight_decay":0.1},
                                {'params': net.backbone.dec_list_GFN[2].q_zxy_mask_generators.parameters(), 'lr': mg_lr_z,"weight_decay":0.1},
                                {'params': net.backbone.ans_projector.parameters(), 'lr': mg_lr_z,"weight_decay":0.1},
                                {'params': net.backbone.LogZ_unconditional, 'lr': z_lr,"weight_decay":0.1}]
            q_zxy_optimizer = optimizer.Adam(q_zxy_param_list)


        loss_sum = 0
        named_params = list(net.named_parameters())
        grad_norm = np.zeros(len(named_params))

        # Define multi-thread dataloader
        if self.HP.SHUFFLE_MODE in ['external']:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.HP.BATCH_SIZE,
                shuffle=False,
                num_workers=self.HP.NUM_WORKERS,
                pin_memory=self.HP.PIN_MEM,
                drop_last=True
            )
        else:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.HP.BATCH_SIZE,
                shuffle=True,
                num_workers=self.HP.NUM_WORKERS,
                pin_memory=self.HP.PIN_MEM,
                drop_last=True
            )

        # Training script
        for epoch in range(start_epoch, self.HP.MAX_EPOCH):

            # Save log information
            logfile = open(
                self.HP.LOG_PATH +
                'log_run_' + self.HP.VERSION+"_seed_"+str(self.HP.SEED) + '.txt',
                'a+'
            )
            logfile.write(
                'nowTime: ' +
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                '\n'
            )
            logfile.close()

            # Learning Rate Decay
            if epoch in self.HP.LR_DECAY_LIST:
                adjust_lr(optim, self.HP.LR_DECAY_R)

            # Externally shuffle
            if self.HP.SHUFFLE_MODE == 'external':
                shuffle_list(dataset.ans_list)

            time_start = time.time()
            # Iteration
            for step, (
                    img_feat_iter,
                    ques_ix_iter,
                    ans_iter
            ) in enumerate(dataloader):
                #tic = time.time()
                optim.zero_grad()

                img_feat_iter = img_feat_iter.cuda()
                ques_ix_iter = ques_ix_iter.cuda()
                ans_iter = ans_iter.cuda()

                for accu_step in range(self.HP.GRAD_ACCU_STEPS):

                    sub_img_feat_iter = \
                        img_feat_iter[accu_step * self.HP.SUB_BATCH_SIZE:
                                      (accu_step + 1) * self.HP.SUB_BATCH_SIZE]
                    sub_ques_ix_iter = \
                        ques_ix_iter[accu_step * self.HP.SUB_BATCH_SIZE:
                                     (accu_step + 1) * self.HP.SUB_BATCH_SIZE]
                    sub_ans_iter = \
                        ans_iter[accu_step * self.HP.SUB_BATCH_SIZE:
                                 (accu_step + 1) * self.HP.SUB_BATCH_SIZE]
                    

                    if self.HP.GFlowOut!="none":
                        q_zxy_optimizer.zero_grad()
                        p_zx_optimizer.zero_grad()
                        q_z_optimizer.zero_grad()
                        pred,LogZ_unconditional,LogPF_qz,LogPB_qz,LogPF_BNN,LogPB_BNN,LogPF_qzxy,LogPB_qzxy,Log_pzx,Log_pz = net(
                            sub_img_feat_iter,
                            sub_ques_ix_iter,
                            sub_ans_iter
                        )
                        loss = loss_fn_keep(pred, sub_ans_iter).sum(1)
                        # only mean-reduction needs be divided by grad_accu_steps
                        # removing this line wouldn't change our results because the speciality of Adam optimizer,
                        # but would be necessary if you use SGD optimizer.
                        # loss /= self.__C.GRAD_ACCU_STEPS
              
                        ####GFN loss
                        LL=-loss
        

                        beta=1
                        N=214354
                        LogR_unconditional=beta*N*LL.detach().clone()+Log_pz.detach().clone()
                        GFN_loss_unconditional=(LogZ_unconditional+LogPF_qz-LogR_unconditional-LogPB_qz)**2#+kl_weight*kl#jointly train the last layer BNN and the mask generator

                        LogR_conditional=beta*LL.detach().clone()+Log_pzx.detach().clone()
                        GFN_loss_conditional=(LogZ_unconditional+LogPF_qzxy-LogR_conditional-LogPB_qzxy)**2#+kl_weight*kl#jointly train the last layer BNN and the mask generator

                        loss=loss.sum()
                        loss_sum += loss.cpu().data.numpy() * self.HP.GRAD_ACCU_STEPS
                        
                        if self.HP.GFlowOut=="bottomup":
                            GFN_loss_conditional.sum().backward(retain_graph=True)
                            pzx_loss=-Log_pzx
                            pzx_loss.sum().backward(retain_graph=True)
                            loss.backward(retain_graph=True)

                            q_zxy_optimizer.step()

                            p_zx_optimizer.step()
                    else:
                        if self.HP.ARM and self.HP.dp_type and self.HP.ctype != "Gaussian":
                            self.forward_mode(True)

                        if self.HP.add_noise:
                            gaussian_noise = np.random.normal(size=sub_img_feat_iter.size()) * self.HP.noise_scalar
                            gaussian_noise = torch.from_numpy(gaussian_noise).type_as(sub_img_feat_iter).cuda()
                            sub_img_feat_iter = sub_img_feat_iter + gaussian_noise


                        pred = net(
                            sub_img_feat_iter,
                            sub_ques_ix_iter
                        )

                        loss = loss_fn(pred, sub_ans_iter)
                        if self.HP.ARM and self.HP.dp_type and self.HP.ctype != "Gaussian":
                            loss_keep = loss_fn_keep(pred, sub_ans_iter).sum(1)
                            penalty = 0
                            prior_sum = 0
                            for layer in self.dropout_list:
                                nll_shape = len(layer.post_nll_true.shape)
                                penalty = penalty + layer.post_nll_true.mean(tuple(np.arange(1, nll_shape))).data - \
                                          layer.prior_nll_true.mean(tuple(np.arange(1, nll_shape))).data
                                prior_sum = prior_sum + layer.prior_nll_true.mean(tuple(np.arange(1, nll_shape)))
                            if self.HP.learn_prior:
                                prior_sum.mean().backward(retain_graph=True)
                            f2 = loss_keep.data - penalty
                            self.forward_mode(False)
                            pred = net(
                                sub_img_feat_iter,
                                sub_ques_ix_iter
                            )
                            loss_keep = loss_fn_keep(pred, sub_ans_iter).sum(1)
                            penalty = 0
                            for layer in self.dropout_list:
                                nll_shape = len(layer.post_nll_true.shape)
                                penalty = penalty + layer.post_nll_true.mean(tuple(np.arange(1, nll_shape))).data - \
                                          layer.prior_nll_true.mean(tuple(np.arange(1, nll_shape))).data
                            f1 = loss_keep.data - penalty#.data
                            self.update_phi_gradient(f1, f2)

                        loss /= self.HP.GRAD_ACCU_STEPS
                        # loss.backward(retain_graph=True)
                        loss.backward()
                        loss_sum += loss.cpu().data.numpy() * self.HP.GRAD_ACCU_STEPS

                    if self.HP.VERBOSE:
                        if dataset_eval is not None:
                            mode_str = self.HP.SPLIT['train'] + '->' + self.HP.SPLIT['val']
                        else:
                            mode_str = self.HP.SPLIT['train'] + '->' + self.HP.SPLIT['test']

                        if step % 1000 == 0:
                            print("\r[version %s][epoch %2d][step %4d/%4d][%s] loss: %.4f, lr: %.2e" % (
                                self.HP.VERSION,
                                epoch + 1,
                                step,
                                int(data_size / self.HP.BATCH_SIZE),
                                mode_str,
                                loss.cpu().data.numpy() / self.HP.SUB_BATCH_SIZE,
                                optim._rate
                            ), end='          ')

                # Gradient norm clipping
                if self.HP.GRAD_NORM_CLIP > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
                        self.HP.GRAD_NORM_CLIP
                    )

                # Save the gradient information
                for name in range(len(named_params)):
                    norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                        if named_params[name][1].grad is not None else 0
                    grad_norm[name] += norm_v * self.HP.GRAD_ACCU_STEPS
                    # print('Param %-3s Name %-80s Grad_Norm %-20s'%
                    #       (str(grad_wt),
                    #        params[grad_wt][0],
                    #        str(norm_v)))
                #print('one iter time:', time.time() - tic)
                optim.step()

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end-time_start)))

            # print('')
            epoch_finish = epoch + 1

            # Save checkpoint
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base
            }
            torch.save(
                state,
                self.HP.CKPTS_PATH +
                'ckpt_' + self.HP.VERSION +"_seed_"+str(self.HP.SEED)+
                '/epoch' + str(epoch_finish) +
                '.pkl'
            )

            # Logging
            logfile = open(
                self.HP.LOG_PATH +
                'log_run_' + self.HP.VERSION +"_seed_"+str(self.HP.SEED)+ '.txt',
                'a+'
            )
            logfile.write(
                'epoch = ' + str(epoch_finish) +
                '  loss = ' + str(loss_sum / data_size) +
                '\n' +
                'lr = ' + str(optim._rate) +
                '\n\n'
            )
            logfile.close()

            # Eval after every epoch
            if dataset_eval is not None and (epoch+1) % 13 == 0:
                self.eval(
                    dataset_eval,
                    state_dict=net.state_dict(),
                    valid=True
                )

            # if self.HP.VERBOSE:
            #     logfile = open(
            #         self.HP.LOG_PATH +
            #         'log_run_' + self.HP.VERSION + '.txt',
            #         'a+'
            #     )
            #     for name in range(len(named_params)):
            #         logfile.write(
            #             'Param %-3s Name %-80s Grad_Norm %-25s\n' % (
            #                 str(name),
            #                 named_params[name][0],
            #                 str(grad_norm[name] / data_size * self.HP.BATCH_SIZE)
            #             )
            #         )
            #     logfile.write('\n')
            #     logfile.close()

            loss_sum = 0
            grad_norm = np.zeros(len(named_params))


    # Evaluation
    def eval(self, dataset, state_dict=None, valid=False):
        setup_seed_2(1)
        data_size = 443757

        elbo_list = []
        self.HP.data_size = data_size
        # Load parameters
        if self.HP.CKPT_PATH is not None:
            print('Warning: you are now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')

            path = self.HP.CKPT_PATH
        else:
            path = self.HP.CKPTS_PATH + \
                   'ckpt_' + self.HP.CKPT_VERSION +"_seed_"+str(self.HP.SEED)+ \
                   '/epoch' + str(self.HP.CKPT_EPOCH) + '.pkl'

        val_ckpt_flag = False
        if state_dict is None:
            val_ckpt_flag = True
            print('Loading ckpt {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            print('Finish!')

        # Store the prediction list
        qid_list = [ques['question_id'] for ques in dataset.ques_list]
        ans_ix_list = []
        pred_list = []
        p_value_list = []

        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        net = Net(
            self.HP,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.eval()

        self.dropout_list = []
        self.dropout_list = self.dropout_list + [enc.mhatt.dropout for enc in net.backbone.enc_list]
        self.dropout_list = self.dropout_list +([enc.ffn.mlp.fc.dropout for enc in net.backbone.enc_list])
        self.dropout_list = self.dropout_list +([enc.dropout1 for enc in net.backbone.enc_list])
        self.dropout_list = self.dropout_list +([enc.dropout2 for enc in net.backbone.enc_list])
        self.dropout_list = self.dropout_list +([dec.mhatt1.dropout for dec in net.backbone.dec_list])
        self.dropout_list = self.dropout_list +([dec.mhatt2.dropout for dec in net.backbone.dec_list])
        self.dropout_list = self.dropout_list +([dec.ffn.mlp.fc.dropout for dec in net.backbone.dec_list])
        self.dropout_list = self.dropout_list +([dec.dropout1 for dec in net.backbone.dec_list])
        self.dropout_list = self.dropout_list +([dec.dropout2 for dec in net.backbone.dec_list])
        self.dropout_list = self.dropout_list +([dec.dropout3 for dec in net.backbone.dec_list])
        self.dropout_list = self.dropout_list +([net.attflat_img.mlp.fc.dropout, net.attflat_lang.mlp.fc.dropout])
        

        if self.HP.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.HP.DEVICES)

        net.load_state_dict(state_dict)

        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.HP.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.HP.NUM_WORKERS,
            pin_memory=True
        )
        loss_fn = torch.nn.BCELoss(reduction='none').cuda()

        label_tensors = torch.zeros([0], dtype=torch.float32)
        sigmoid_tensors = torch.zeros([0], dtype=torch.float32)
        for step, (
                img_feat_iter,
                ques_ix_iter,
                ans_iter
        ) in enumerate(dataloader):

            print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(data_size / self.HP.EVAL_BATCH_SIZE),
            ), end='          ')

            img_feat_iter = img_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()

            if self.HP.add_noise:
                gaussian_noise = np.random.normal(size=img_feat_iter.size()) * self.HP.noise_scalar
                gaussian_noise = torch.from_numpy(gaussian_noise).type_as(img_feat_iter).cuda()
                img_feat_iter = img_feat_iter + gaussian_noise
            



            if self.HP.GFlowOut!="none":
        
                pred,LogZ_unconditional,LogPF_qz,LogPB_qz,LogPF_BNN,LogPB_BNN,LogPF_qzxy,LogPB_qzxy,Log_pzx,Log_pz = net(
                    img_feat_iter,
                    ques_ix_iter,
                    ans_iter.cuda()
                )
            else:
                pred = net(
                    img_feat_iter,
                    ques_ix_iter
                )
            sigmoid_tensors = torch.cat((sigmoid_tensors.cpu(), pred.detach().cpu()), 0)
            label_tensors = torch.cat((label_tensors.cpu(), ans_iter.cpu()), 0)

            pred_np = pred.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)
            
            pred_uncertain = torch.zeros([0]).cuda()
            for iii in range(self.HP.uncertainty_sample):

                if self.HP.GFlowOut!="none":
                    pred,LogZ_unconditional,LogPF_qz,LogPB_qz,LogPF_BNN,LogPB_BNN,LogPF_qzxy,LogPB_qzxy,Log_pzx,Log_pz = net(
                        img_feat_iter,
                        ques_ix_iter,
                        ans_iter.cuda()
                    )
                    pred=pred.data
                else:
                    pred = net(img_feat_iter, ques_ix_iter).data
                # print('shape', pred.shape, ans_iter.shape)
                loss = loss_fn(pred, ans_iter.cuda()).sum(1)
                penalty = 0
                if (self.HP.dp_type or self.HP.concretedp) and self.HP.GFlowOut=="none":
                    for layer in self.dropout_list:
                        nll_shape = len(layer.post_nll_true.shape)
                        penalty = penalty + layer.post_nll_true.mean(tuple(np.arange(1, nll_shape))).data - \
                                  layer.prior_nll_true.mean(tuple(np.arange(1, nll_shape))).data
                    if self.HP.dp_type:
                        elbo_list.append((-loss.cpu().data + penalty.cpu()).mean())
                    else:
                        elbo_list.append((-loss.cpu().data + 2. / self.HP.data_size * penalty.cpu()).mean())
                else:
                    elbo_list.append((-loss.cpu().data).mean())
                pred_uncertain = torch.cat([pred_uncertain, pred.unsqueeze(2)], 2)
            net.eval()
            # with open(os.path.join(self.HP.LOG_PATH +
            #                        'log_run_' + self.HP.CKPT_VERSION + 'prob.pkl', ), 'wb') as f:
            #     cPickle.dump(pred_uncertain.cpu(), f)
            # break
            p_value = np.squeeze(two_sample_test_batch(pred_uncertain, self.HP.uncertainty_sample)) # sample, batch, class
            # Save the answer index
            if pred_argmax.shape[0] != self.HP.EVAL_BATCH_SIZE:
                pred_argmax = np.pad(
                    pred_argmax,
                    (0, self.HP.EVAL_BATCH_SIZE - pred_argmax.shape[0]),
                    mode='constant',
                    constant_values=-1
                )
            if p_value.shape[0] != self.HP.EVAL_BATCH_SIZE:
                p_value = np.pad(
                    p_value,
                    (0, self.HP.EVAL_BATCH_SIZE - p_value.shape[0]),
                    mode='constant',
                    constant_values=-1
                )
            ans_ix_list.append(pred_argmax)
            p_value_list.append(p_value)

            # Save the whole prediction vector
            if self.HP.TEST_SAVE_PRED:
                if pred_np.shape[0] != self.HP.EVAL_BATCH_SIZE:
                    pred_np = np.pad(
                        pred_np,
                        ((0, self.HP.EVAL_BATCH_SIZE - pred_np.shape[0]), (0, 0)),
                        mode='constant',
                        constant_values=-1
                    )

                pred_list.append(pred_np)

        ece = ECELoss(n_bins = 10)(sigmoid_tensors, label_tensors)
        print('')
        print('ELBO***************************', np.mean(elbo_list)*100)
        ans_ix_list = np.array(ans_ix_list).reshape(-1)
        p_value_list = np.array(p_value_list).reshape(-1)

        result = [{
            'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],  # ix_to_ans(load with json) keys are type of string
            'question_id': int(qid_list[qix]),
            'p_value': float(p_value_list[qix])
        }for qix in range(qid_list.__len__())]

        # Write the results to result file
        if valid:
            if val_ckpt_flag:
                result_eval_file = \
                    self.HP.CACHE_PATH + \
                    'result_run_' + self.HP.CKPT_VERSION +"_seed_"+str(self.HP.SEED)+ \
                    '.json'
            else:
                result_eval_file = \
                    self.HP.CACHE_PATH + \
                    'result_run_' + self.HP.VERSION +"_seed_"+str(self.HP.SEED)+ \
                    '.json'

        else:
            if self.HP.CKPT_PATH is not None:
                result_eval_file = \
                    self.HP.RESULT_PATH + \
                    'result_run_' + self.HP.CKPT_VERSION+"_seed_"+str(self.HP.SEED) + \
                    '.json'
            else:
                result_eval_file = \
                    self.HP.RESULT_PATH + \
                    'result_run_' + self.HP.CKPT_VERSION + \
                    '_epoch' + str(self.HP.CKPT_EPOCH) +"_seed_"+str(self.HP.SEED)+ \
                    '.json'

            print('Save the result to file: {}'.format(result_eval_file))

        json.dump(result, open(result_eval_file, 'w'))

        # Save the whole prediction vector
        if self.HP.TEST_SAVE_PRED:

            if self.HP.CKPT_PATH is not None:
                ensemble_file = \
                    self.HP.PRED_PATH + \
                    'result_run_' + self.HP.CKPT_VERSION +"_seed_"+str(self.HP.SEED)+ \
                    '.json'
            else:
                ensemble_file = \
                    self.HP.PRED_PATH + \
                    'result_run_' + self.HP.CKPT_VERSION + \
                    '_epoch' + str(self.HP.CKPT_EPOCH) +"_seed_"+str(self.HP.SEED)+ \
                    '.json'

            print('Save the prediction vector to file: {}'.format(ensemble_file))

            pred_list = np.array(pred_list).reshape(-1, ans_size)
            result_pred = [{
                'pred': pred_list[qix],
                'question_id': int(qid_list[qix])
            }for qix in range(qid_list.__len__())]

            pickle.dump(result_pred, open(ensemble_file, 'wb+'), protocol=-1)


        # Run validation script
        if valid:
            # create vqa object and vqaRes object
            ques_file_path = self.HP.QUESTION_PATH['val']
            ans_file_path = self.HP.ANSWER_PATH['val']

            vqa = VQA(ans_file_path, ques_file_path)
            vqaRes = vqa.loadRes(result_eval_file, ques_file_path)

            # create vqaEval object by taking vqa and vqaRes
            vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

            # evaluate results
            """
            If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
            By default it uses all the question ids in annotation file
            """
            uncertainty_result = vqaEval.evaluate(qid_list)

            # print accuracies
            print("\n")
            print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            # print("Per Question Type Accuracy is the following:")
            # for quesType in vqaEval.accuracy['perQuestionType']:
            #     print("%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType]))
            # print("\n")
            print("Per Answer Type Accuracy is the following:")
            for ansType in vqaEval.accuracy['perAnswerType']:
                print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            print("\n")
            print("Overall uncertainty is: %.02f, %.02f, %.02f,\n" % (vqaEval.uncertainty['overall'][0],
                                                                      vqaEval.uncertainty['overall'][1],
                                                                      vqaEval.uncertainty['overall'][2],))
            # print("Per Question Type Accuracy is the following:")
            # for quesType in vqaEval.accuracy['perQuestionType']:
            #     print("%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType]))
            # print("\n")
            print("Per Answer Type Uncertainty is the following:")
            for ansType in vqaEval.uncertainty['perAnswerType']:
                print("%s : %.02f, %.02f, %.02f," % (ansType, vqaEval.uncertainty['perAnswerType'][ansType][0],
                                                     vqaEval.uncertainty['perAnswerType'][ansType][1],
                                                     vqaEval.uncertainty['perAnswerType'][ansType][2],))
            print("\n")

            print("ECE for this model is the following:")
            print("%.04f"%(ece[0]))
            if val_ckpt_flag:
                print('Write to log file: {}'.format(
                    self.HP.LOG_PATH +
                    'log_run_' + self.HP.CKPT_VERSION +"_seed_"+str(self.HP.SEED)+ '.txt',
                    'a+')
                )

                logfile = open(
                    self.HP.LOG_PATH +
                    'log_run_' + self.HP.CKPT_VERSION +"_seed_"+str(self.HP.SEED)+ '.txt',
                    'a+'
                )

                with open(os.path.join(self.HP.LOG_PATH +
                    'log_run_' + self.HP.CKPT_VERSION +"_seed_"+str(self.HP.SEED)+ 'uc.pkl',), 'wb') as f:
                    cPickle.dump(uncertainty_result, f)

            else:
                print('Write to log file: {}'.format(
                    self.HP.LOG_PATH +
                    'log_run_' + self.HP.VERSION +"_seed_"+str(self.HP.SEED)+ '.txt',
                    'a+')
                )

                logfile = open(
                    self.HP.LOG_PATH +
                    'log_run_' + self.HP.VERSION + '.txt',
                    'a+'
                )

                with open(os.path.join(self.HP.LOG_PATH +
                    'log_run_' + self.HP.VERSION +"_seed_"+str(self.HP.SEED) + 'uc.pkl',), 'wb') as f:
                    cPickle.dump(uncertainty_result, f)

            logfile.write("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            for ansType in vqaEval.accuracy['perAnswerType']:
                logfile.write("%s : %.02f " % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            logfile.write("\n")
            logfile.write("Overall uncertainty is: %.02f, %.02f, %.02f\n" % (vqaEval.uncertainty['overall'][0],
                                                                             vqaEval.uncertainty['overall'][1],
                                                                             vqaEval.uncertainty['overall'][2]))
            for ansType in vqaEval.uncertainty['perAnswerType']:
                logfile.write("%s : %.02f, %.02f, %.02f\n" % (ansType, vqaEval.uncertainty['perAnswerType'][ansType][0],
                                                              vqaEval.uncertainty['perAnswerType'][ansType][1],
                                                              vqaEval.uncertainty['perAnswerType'][ansType][2],))
            logfile.write("\n")
            logfile.write("%.04f"%(ece[0]))
            logfile.write("\n")
            logfile.close()

            #save as csv
            import csv
            ToSave=[str(self.HP.VERSION),str(self.HP.add_noise),str(vqaEval.accuracy['overall'])]
            
            for ansType in vqaEval.accuracy['perAnswerType']:
                ToSave.append(str(vqaEval.accuracy['perAnswerType'][ansType]))

            with open('VQAresults.csv','a') as fd:
                writer = csv.writer(fd)
                #fd.write(ToSave)
                writer.writerow(ToSave)

    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.HP.VERSION)
            self.train(self.dataset, self.dataset_eval)

        elif run_mode == 'val':
            self.eval(self.dataset, valid=True)

        elif run_mode == 'test':
            self.eval(self.dataset)

        else:
            exit(-1)


    def empty_log(self, version):
        print('Initializing log file ........')
        if (os.path.exists(self.HP.LOG_PATH + 'log_run_' + version+"_seed_"+str(self.HP.SEED) + '.txt')):
            os.remove(self.HP.LOG_PATH + 'log_run_' + version+"_seed_"+str(self.HP.SEED) + '.txt')
        print('Finished!')
        print('')


    def update_phi_gradient(self, f1, f2):
        self.net.attflat_img.mlp.fc.dropout.update_phi_gradient(f1, f2)
        self.net.attflat_lang.mlp.fc.dropout.update_phi_gradient(f1, f2)

        # for layer in self.dropout_list:
        #     layer.update_phi_gradient(f1, f2)


    def forward_mode(self, mode):
        for layer in self.dropout_list:
            layer.forward_mode = mode



def two_sample_test_batch(prob, sample_num):
    probmean = torch.mean(prob,2)
    values, indices = torch.topk(probmean, 2, dim=1)
    aa = prob.gather(1, indices[:,0].unsqueeze(1).unsqueeze(1).repeat(1,1,sample_num))
    bb = prob.gather(1, indices[:,1].unsqueeze(1).unsqueeze(1).repeat(1,1,sample_num))
    pvalue = sts.ttest_ind(aa.cpu(),bb.cpu(), axis=2, equal_var=False).pvalue
    return pvalue