import argparse, textwrap

formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=50)

def cla():
    parser = argparse.ArgumentParser(description='list of arguments',formatter_class=formatter)

    parser.add_argument('--datadir'   , type=str, required=True, help=textwrap.dedent('''Dataset directory path'''))
    parser.add_argument('--n_train'   , type=int, default=4000, help=textwrap.dedent('''Number of training samples to use. Cannot be more than that available.'''))
    #parser.add_argument('--g_type'    , type=str, required=True, choices=['type1','type2'],help=textwrap.dedent('''type1: Generator can use latent space dimension, with the latent
                                                                                                                #variable injected at every layer via conditional instance normalization.\n 
                                                                                                                #type2: The latent variable is injected only in the generator input as an 
                                                                                                                #additional input channel, with batch normalization in the rest of the network.'''))
    parser.add_argument('--reg_param' , type=float, default=1e-7, help=textwrap.dedent('''Regularization parameter'''))
    parser.add_argument('--act_param' , type=float, default=0.2, help=textwrap.dedent('''Activation parameter'''))
    parser.add_argument('--gp_coef'   , type=float, default=10.0, help=textwrap.dedent('''Gradient penalty parameter'''))
    parser.add_argument('--n_critic'  , type=int, default=4, help=textwrap.dedent('''Number of critic updates per generator update'''))
    parser.add_argument('--n_epoch'   , type=int, default=1000, help=textwrap.dedent('''Maximum number of epochs'''))
    parser.add_argument('--n_epochfolder'   , type=int, default=500, help=textwrap.dedent('''Maximum number of epochs'''))
 
    parser.add_argument('--z_dim'     , type=int, default=1, help=textwrap.dedent('''Dimension of the latent variable, when using type1 C-GAN'''))
    parser.add_argument('--batch_size', type=int, default=16, help=textwrap.dedent('''Batch size while training'''))
    parser.add_argument('--LAMBDA', type=int, default=100, help=textwrap.dedent('''L1 loss LAMBDA'''))

    parser.add_argument('--seed_no'      , type=int, default=1008, help=textwrap.dedent('''Fix the random seed'''))
    parser.add_argument('--savefig_freq' , type=int, default=100, help=textwrap.dedent('''Number of epochs after which a snapshot and plots are saved'''))
    parser.add_argument('--evalmet_freq' , type=int, default=100, help=textwrap.dedent('''Number of epochs after which a tthe validation metrix is evaluated'''))
    parser.add_argument('--save_suffix'  , type=str, default='', help=textwrap.dedent('''Suffix to directory where trained network/results are saved'''))
    parser.add_argument('--MC_samples'   , type=int, default=400, help=textwrap.dedent('''Number of samples used to generate emperical statistics'''))
    parser.add_argument('--sample_plots' , type=int, default=5, help=textwrap.dedent('''Number of validation samples used to generate plots'''))  #for how many Ys you generate zs (for same Y) to get plots 
    

    parser.add_argument('--data_suffix'  , type=str, default='', help=textwrap.dedent('''Suffix to test results directory where results on test data are saved'''))
    parser.add_argument('--main_dir'     , type=str, default='exps', help=textwrap.dedent('''Parent directory saving the various versions of trained networks'''))
    parser.add_argument('--checkpoint_id', type=int, default=-1, help=textwrap.dedent('''The checkpoint index to load when testing'''))
    parser.add_argument('--checkpoint_retrain', type=int, default=0, help=textwrap.dedent('''The checkpoint index to retrain from'''))

    parser.add_argument('--Etype' , type=float, default=0, help=textwrap.dedent('''Add output or no'''))
    parser.add_argument('--ADDtanh' , type=float, default=1, help=textwrap.dedent('''Add tanh at last layer or no'''))
    
    
    parser.add_argument('--patient_id', type=int, default=0, help=textwrap.dedent('''Sample point needed to test'''))
    return parser.parse_args()


