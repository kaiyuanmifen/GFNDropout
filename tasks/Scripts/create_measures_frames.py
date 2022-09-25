from six.moves import cPickle
import os
import pandas as pd
from ast import literal_eval

base_path_ckpt = "/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/"
testing_predictions = pd.read_csv("/home/mila/b/bonaventure.dossou/GFNDropout/tasks/Scripts/testresult.csv", header=None)

methods_saving_points = [x[0] for x in os.walk(base_path_ckpt)][1:]
methods_names_list, ood_f1_list, ood_precision_list, ood_recall_list, ood_bal_acc_list = [], [], [], [], []
for method in methods_saving_points:
	try:
		method_name = method.split('/')[-1]
		method_reference_model = os.path.join(method, 'best.model')
		method_test_details = testing_predictions[testing_predictions[0]==method_reference_model].values.tolist()[0]
		with open(os.path.join(method, 'histories_' + '.pkl'), 'rb') as f:
			histories = cPickle.load(f)
		# val_accuracy_history = histories.get('val_accuracy_hisotry', {})
		accuracy_list = []
		for _, values in histories['val_accuracy_history'].items():
			accuracy_list.append(values['accuracy'])
		
		methods_names_list.append(method_name)
		# iid_acc_list.append(sum(accuracy_list)/len(accuracy_list))
		metrics = literal_eval(method_test_details[5])
		ood_f1_list.append(metrics[0])
		ood_precision_list.append(metrics[1])
		ood_recall_list.append(metrics[2])
		ood_bal_acc_list.append(metrics[3])

	except Exception as e:
		pass

frame =  pd.DataFrame()
frame['Method'] = methods_names_list
frame['ood_macro_f1'] = ood_f1_list
frame['ood_precision_list'] = ood_precision_list
frame['ood_recall_list'] = ood_recall_list
frame['ood_bal_acc_list'] = ood_bal_acc_list

frame.sort_values(by=['ood_recall_list'], ascending=False, inplace=True)
frame.to_csv('/home/mila/b/bonaventure.dossou/GFNDropout/tasks/Scripts/{}.csv'.format('methods_iid_ood_performance'), index=False)

print(frame)