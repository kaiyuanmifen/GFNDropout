python -u ../image_classification/main.py train \
										--model=MLP_GFN \
										--GFN_dropout False \
										--dataset=iecu \
										--lambas=.001 \
										--optimizer=momentum \
										--lr=0.1 \
										--schedule_milestone="[60, 120]" \
										--add_noisedata=False \
										--concretedp True \
										--dptype False \
										--fixdistrdp False \
										--mask "none" \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_IECU_MLP_Concrete" \
										--max_epoch 100 \


python -u ../image_classification/main.py train \
										--model=MLP_GFN \
										--GFN_dropout False \
										--dataset=iecu \
										--lambas=.001 \
										--optimizer=momentum \
										--lr=0.1 \
										--schedule_milestone="[60, 120]" \
										--add_noisedata=False \
										--dptype True \
										--mask "none" \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_IECU_MLP_Contextual" \
										--max_epoch 100 \


python -u ../image_classification/main.py train \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=iecu \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "random" \
										--BNN False \
										--model_name "_IECU_MLP_GFN_Random" \
										--beta 0.001 \
										--max_epoch 100 \
										

python -u ../image_classification/main.py train \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=iecu \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "none" \
										--BNN False \
										--model_name "_IECU_MLP_GFN_None" \
										--beta 0.001 \
										--max_epoch 100 \



python -u ../image_classification/main.py train \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=iecu \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "topdown" \
										--BNN False \
										--model_name "_IECU_MLP_GFN_TopDown" \
										--beta 0.001 \
										--max_epoch 100 \


python -u ../image_classification/main.py train \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=iecu \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "bottomup" \
										--BNN False \
										--model_name "_IECU_MLP_GFN_BottomUp" \
										--beta 0.001 \
										--max_epoch 100 \

python -u ../image_classification/main.py train \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=iecu \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "upNdown" \
										--BNN False \
										--model_name "_IECU_MLP_GFN_UpDown" \
										--beta 0.001 \
										--max_epoch 100 \