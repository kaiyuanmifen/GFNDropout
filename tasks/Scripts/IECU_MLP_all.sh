python -u ../image_classification/main.py train \
										--model=ARMMLP \
										--GFFN_dropout False \
										--dataset=iecu \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--is_iecu True \
										--concretedp True \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_IECU_ARMMLP_Concrete" \
										--max_epoch 100 \


python -u ../image_classification/main.py train \
										--model=ARMMLP \
										--GFFN_dropout False \
										--dataset=iecu \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype True \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_IECU_ARMMLP_Contextual" \
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
										--is_iecu True \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "random" \
										--BNN False \
										--model_name "_IECU_MLP_GFN" \
										--beta 0.001 \
										--max_epoch 100 \
										

python -u ../image_classification/main.py train \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=iecu \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--is_iecu True \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "none" \
										--BNN False \
										--model_name "_IECU_MLP_GFN" \
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
										--is_iecu True \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "topdown" \
										--BNN False \
										--model_name "_IECU_MLP_GFN" \
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
										--is_iecu True \
										--dropout_distribution 'bernoulli' \
										--mask "bottomup" \
										--BNN False \
										--model_name "_IECU_MLP_GFN" \
										--beta 0.001 \
										--max_epoch 100 \

python -u ../image_classification/main.py train \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=iecu \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--is_iecu True \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "upNdown" \
										--BNN False \
										--model_name "_IECU_MLP_GFN" \
										--beta 0.001 \
										--max_epoch 100 \