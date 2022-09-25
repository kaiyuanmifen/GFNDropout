from six.moves import cPickle
import os
import pandas as pd
import matplotlib.pyplot as plt

base_path_ckpt = "/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/"
testing_predictions = pd.read_csv("/home/mila/b/bonaventure.dossou/GFNDropout/tasks/Scripts/testresult.csv", header=None)

methods_saving_points = [x[0] for x in os.walk(base_path_ckpt)][1:]
methods_names_list, iid_acc_list, ood_acc_list = [], [], []
for method in methods_saving_points:
	try:	
		method_name = method.split('/')[-1]
		method_reference_model = os.path.join(method, 'best.model')
		method_test_details = testing_predictions[testing_predictions[0]==method_reference_model].values.tolist()[0]
		with open(os.path.join(method, 'histories_' + '.pkl'), 'rb') as f:
			histories = cPickle.load(f)

		val_accuracy_history = histories['val_accuracy_history']
		val_loss_list = []
		for _, values in histories['val_accuracy_history'].items():
			val_loss_list.append(values['ece'])

		plt.plot([i for i in range(1, 101)], val_loss_list, label=method_name)
	except:
		pass

plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend(loc='best')
plt.savefig('/home/mila/b/bonaventure.dossou/GFNDropout/tasks/Scripts/val_loss_plot_gflowout.png')