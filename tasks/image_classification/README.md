# Contextual dropout

This repository contains the code for contextual dropout for Bayesian neural networks. The code requires GPU computation. 

## Requirements
    pytorch>1.0.0
    tnt
    fire
    tqdm
    numpy
    tensorboardX

## Configuration
For each model, please change the config.py file and set the model flag to false except for the model that you want to run.
For example to run Bernoulli Contextual dropout
In the config file set the flags below.

    ##MC
    dptype = False
    fixdistrdp = True
    dropout_distribution = 'bernoulli' ##Whether to use bernoulli or gaussian (Please be careful with capitalization)

    ##Concrete
    dptype = False
    concretedp = True

    ##Contextual
    dptype = True
    ctype = "Bernoulli" ##Whether to use Bernoulli or Guassian dropout

After setting the correct configuration, please run the following commands to execute the main file.
## Usage
    python main.py <function> [--args=value]
        <function> := train | test | help
    example MNIST:
        python main.py train --model=ARMMLP --dataset=mnist --lambas='[.0,.0,.0,.0]' --optimizer=adam --lr=0.001 --add_noisedata=False
    example CIFAR:
        python main.py train --model=ARMWideResNet --dataset=cifar10 --lambas=.001 --optimizer=momentum --lr=0.1 --schedule_milestone="[60, 120]" --add_noisedata=False
        

