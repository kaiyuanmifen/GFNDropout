from six.moves import cPickle
import os
import pandas as pd
import matplotlib.pyplot as plt

base_path_ckpt = "/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/"

all_methods_checkpoint = ['ARMMLP_AL_IECU_ARMMLP_Concrete_20221010010701_{}/']

methods_names_list, train_loss, val_loss = [], [], []
for method in all_methods_checkpoint:
	try:	
		method_name = '_'.join(method.split('/')[0].split('_')[:-2])
		all_all_val_losses = []
		for al_round in range(6):
			method_reference_model = os.path.join(base_path_ckpt, method.format(al_round), 'best.model')
			with open(os.path.join(base_path_ckpt, method.format(al_round), 'histories_' + '.pkl'), 'rb') as f:
				histories = cPickle.load(f)
			val_accuracy_history = histories['val_accuracy_history']
			val_loss_list = []
			for _, values in histories['val_accuracy_history'].items():
				val_loss_list.append(values['ece'])
			al_round_loss = sum(val_loss_list)/len(val_loss_list)
			all_all_val_losses.append(al_round_loss)
		plt.plot([i for i in range(6)], all_all_val_losses, label=method_name)
	except:
		pass

plt.xlabel('Epochs')
plt.ylabel('AL Validation Loss')
plt.legend(loc='best')
plt.savefig('/home/mila/b/bonaventure.dossou/GFNDropout/tasks/Scripts/al_val_loss_plot_gflowout.png')