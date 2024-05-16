#!/usr/bin/env python
import sys

import flatquad_landing_experiment as exp
from util import generate_run_commands, generate_base_command, dict_permutations, available_gpus

PROJECT_NAME = 'flatquad'

flatquad_configs = {
    # 'T_value_target': [1/2, 1, 2.],
    # 'weight_decay': [.0001, .0005, .001, .005, .01, .05, .1],
    # 'weight_decay': [.0002, .0003, .0004, .0005, .0007, .0009, .001, .0012, .0015],
    # 'nn_type': ['minout_softplus', 'softplus', 'experimental'],
    # 'seed': [1,2,3,4,5,6,7,8]
    # 'lr_init': [0.05, 0.02, 0.01, 0.005],
    # 'lr_final': [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001],
    # 'nn_N_epochs': [1024, 2048],
    # 'nn_warmstart_fraction': [1/4, 1/3, 1/2, 1.],
    # 'L_v': [200, 500],
    # 'L_vx': [300, 400, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000],
    # 'vx_loss_d': [.001, .005, 0.01, .05, .1, .5, 1],
    # 'v_loss_d': [.001, .005, .01, .05, .1, .5, 1],
    # 'vx_loss_d': [0.01, .05, .1, .5],
    # 'v_loss_d': [.01, .05, .1, .5],

    # 'nn_type': ['leaky', 'softplus'],
    # 'thin_data_denominator': [1, 5, 10, 20, 50, 100, 100000],
    'lr_final': [0.0001, .0002, .0005, .001, .002, .005],
    'weight_decay': [.0005, 0.001, 0.002, .005, .01, .02],

    'inv_vx_loss_fadeout': [0.],
    'proposal_strategy': ['max_kernel_adaptive', 'uniform_uncertain', 'uniform_all'],
    
    # 'T_value_target': [0.1, 0.3, 1., 2., 3.],
    # 'proposal_kernel_scaling': [0.1, 1.],
    # 'include_future_data': ['True', 'False'],
    # 'thin_data_denominator': [10, 5, 3, 2],


    'vx_loss_d': [ 0.1, 0.3 ], 

    # 'active_learning_batchsize': [32, 64, 128, 256, 512],




    # OUTPUT & VISUALISATION
    # (euler config here so we can keep local debugging type config in main file)
    'wandb': [True],

    'savefigs': [False],
    'wandbfigs': [True],
    'showfigs': [False],

    'ipdb_interval': [0],
}

def main():
    command_list = []
    flags_combinations = dict_permutations(flatquad_configs)

    # shitty argparse :)
    do_print = len(sys.argv) > 1 and sys.argv[1] in ('-p', '--print')

    for flags in flags_combinations:
        cmd = generate_base_command(exp, flags=flags)
        if do_print:
            print(cmd)
        command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=1,
                          mode='euler',
                          duration='3:59:00',
                          prompt=True,
                          mem=8192)


if __name__ == '__main__':
    main()
