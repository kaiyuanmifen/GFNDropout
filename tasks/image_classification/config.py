import warnings


class DefaultConfig:
    """ model: ARMLeNet5 | ARMMLP | ARMWideResNet (default: ARMLeNet5)
        optimizer: adam | momentum (default: adam)
        dataset: mnist | cifar10 | cifar100 (default: mnist)
        lambas: L0 regularization strength (default: [10, 0.5, 0.1, 10])
        ar: use AR if True, else use ARM (default: False)
        hardsigmoid: use hardsigmoid if True, else use sigmoid
        k: the hyper-parameter that controls distribution over gates (default: 7)
        log_dir: directory for Tensorboard log (default: log)
        checkpoints_dir: directory for checkpoints (default: 'checkpoints')
        seed: seed for initializing training (default: None)
        max_epoch: number of total epochs to run (default: 200)
        start_epoch: manual epoch number (useful on restarts)
        use_gpu: use GPU or not (default: True)
        load_file: path to checkpoint (default: '')
        batch_size: mini-batch size (default: 128)
        lr: initial learning rate (default: 0.001)
        lr_decay: learning rate decay (default: 0.2)
        weight_decay: weight decay (default: 5e-4)
        momentum: momentum (default: 0.9)
        schedule_milestone: schedule for learning rate decay (default: [])
        t: threshold for gate. gate = 1 if gate > t; else gate = 0. (default: 0.5)
        use_t_in_training: use binary gate for training if True, else use continuous value (default: False)
        use_t_in_testing: use binary gate for testing if True, else use continuous value (default: True)
        lenet_dr: initial dropout rate for LeNet model (default: 0.5)
        mlp_dr: initial dropout rate for MLP model (default: 0.5)
        wrn_dr: initial dropout rate for WRN model (default: 0.01)
        local_rep: stochastic level (default: True)
        gpus: number of gpus (default: 1)
        note: note shown in log title (default: '')
        verbose: verbose mode. (default: True)
        print_freq: print frequency (default: 100) """

    model = 'ARMMLP_GFN'
    
    
    optimizer = 'adam'
    dataset = 'mnist'
    # lambas = [.1, .1, .1]  #MLP
    lambas = [10, 0.5, 0.1, 10]  # LeNet
    #lambas = 0.1   # WRN
    ar = False
    hardsigmoid = False
    k = 0.01

    # for local run
    # log_dir = './log'
    # checkpoints_dir = './checkpoints'
    # for tacc run xinjie
    # log_dir = '/work/06792/kt22354/maverick2/contextual_dropout/log'
    # checkpoints_dir = '/work/06792/kt22354/maverick2/contextual_dropout/checkpoints'
    # # for tacc run shujian
    log_dir =  '../../log'
    checkpoints_dir = '../../checkpoints'

    GFN_dropout=False
    augment_test=False
    mask="none"
    BNN=False
    beta=1.0

    seed = None
    use_gpu = False
    load_file = ''
    batch_size =128 #default 128
    max_epoch = 200#default 200
    lr = 0.001
    lr_decay = 0.2
    weight_decay = 5e-4
    momentum = 0.9
    schedule_milestone = []
    t = 0.0
    use_t_in_training = False
    lenet_dr = 0.5
    #mlp_dr = 0.5
    mlp_dr = 0.5
    wrn_dr = 0.5
    gpus = 1
    note = ''
    verbose = True
    print_freq = 1

    #flag
    #when run the baseline set dptype = False ; use_t_in_testing = True ; local_rep = True

    #############the below is our local version config
    t_test = True
    model_name = ""
    dptype = False
    N_t = 50 ##Number of epochs before the coefficient of kl loss is 1
    var_dropout = False ##Whether to use variational dropout
    ctype = "Bernoulli" ##Whether to use Guassian or Bernouli dropout
    sd = 0.5
    temp = 0.0001
    eta = 138.6 ## the prior distribution can be possibly learned.
    lambda_kl = 1.0 # the weights for KL term.
    rb = False # whether use rao-blackwellization for KL; For Gumbel: True; For ARM: False
    se = False  #whether use soft dropout
    ## channel wise
    dpcha = True
    cha_factor = 10
    use_t_in_testing = True
    local_rep = False
    sample_num = 20
    test_sample_mode = 'greedy'
    batchtrain = False
    recorddp = False
    learn_prior = True
    learn_prior_scalar = True
    encoder_lr_factor = 1.0
    optim_method = False
    concrete_temp =0.1
    sparse_arm = False
    cluster_penalty = 0.0
    cp_anneal_rate = 0.1
    kl_anneal_rate = 0.0 #MLP only: we only update this for mlp
    valtestseed = 2
    # seedtraintest = None  # train and val use one seed; test use one seed; but these two seed could be the same
    # noise = 0.0
    fixdistrdp = False
    concretedp = False
    concretedp_cha = False
    gumbelconcrete = True
    add_noisedata =False
    labelnoise_train = 0.0
    labelnoise_val = 0.0


    pr_bernoulli = 1.0
    noise_scalar = 1.0
    # gumbelconcrete_kl = 0.5
    shrink = 0.01
    dropout_distribution = 'gaussian'
    use_uniform_mask = False
    #resume the model
    start_model = None
    start_optim = None
    start_epoch = 0
    start_from = None

    finetune = False
    mask_type = 'pi_sum'
    add_pi = False
    pruning = False
    pruingrate = 50

    #the below is the baseline version config
    #dptype = False
    #use_t_in_testing = True
    #local_rep = True
    #sample_num = 20
    #batchtrain = False

def parse(self, kwargs):
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    self.title_note = 'ar={} hs={} wd={} lr={} lambda={} k={} t={} mask={} BNN={}'.format(self.ar, self.hardsigmoid,
                                                                           self.weight_decay, self.lr, self.lambas,
                                                                           self.k, self.t,
                                                                           self.mask,self.BNN)
    str = ''
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))
            str += "{}: {}<br/>".format(k, getattr(self, k))
    return str


DefaultConfig.parse = parse
opt = DefaultConfig()

