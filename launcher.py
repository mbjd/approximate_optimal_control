#!/usr/bin/env python
import sys

import numpy as np

from util import (available_gpus, dict_permutations, random_dict_permutations,
                  generate_base_command, generate_run_commands)

PROJECT_NAME = 'flatquad'
# PROJECT_NAME = 'orbits'

flatquad_configs = {
    # 'T_value_target': [1/2, 1, 2.],
    # 'weight_decay': [.0001, .0005, .001, .005, .01, .05, .1],
    # 'weight_decay': [.0002, .0003, .0004, .0005, .0007, .0009, .001, .0012, .0015],
    # 'nn_type': ['minout_softplus', 'softplus', 'experimental'],
    # 'lr_init': [0.05, 0.02, 0.01, 0.005],
    # 'lr_final': [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001],
    # 'nn_N_epochs': [1024, 2048],
    # 'nn_warmstart_fraction': [1/4, 1/3, 1/2, 1.],
    # 'L_v': [200, 500],
    # 'L_vx': [300, 400, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000],
    # 'vx_loss_d': [.001, .005, 0.01, .05, .1, .5, 1],
    # 'v_loss_d': [.001, .005, .01, .05, .1, .5, 1],
    # 'vx_loss_d': [.1, .2, .5],
    # 'v_loss_d': [.01, .05, .1, .5],

    # 'nn_type': ['leaky', 'softplus'],
    # 'thin_data_denominator': [1, 5, 10, 20, 50, 100, 100000],
    # 'lr_final': [.00005, .0001, .0002, .0005, .001, .002, .005, .01, .02, .05],
    # 'seed': [1,2,3,4,5,6,7,8],
    # 'weight_decay': [.0005, .0007, .001, .002, .005, .007, .01, .02],
    # 'weight_decay': .005, # .001 next?
    # 'vx_loss_d': [ 0.1, 0.2, 0.3, 0.4, 0.5 ],

    # 'inv_vx_loss_fadeout': [0.],
    # 'proposal_strategy': ['max_kernel_adaptive', 'uniform_uncertain', 'uniform_all'],

    # 'T_value_target': [0.1, 0.3, 1.],
    # 'proposal_kernel_scaling': [0.1, 1.],
    # 'include_future_data': ['True', 'False'],
    # 'thin_data_denominator': [10, 5, 3, 2],


    # recreating 'comic-deluge'

    # 'seed': [1,2,3,4,5,6,7,8],
    # 'nn_value_sweep': [True, False],
    # 'lr_final': [0.0002, 0.0005, 0.001, 0.002, 0.005],
    # 'weight_decay': [0.001, 0.002, 0.005, 0.01],
    # 'lr_final': np.logspace(-4, -2, 8),
    # 'vx_loss_d': [0.3],
    # 'inv_vx_loss_fadeout': [0.],


    # ... and similar ones
    'nn_value_sweep': [True, False],
    # 'nn_layer_dim': [64, 128, 256],
    'nn_layer_dim': [128, 256],
    # 'thin_data': [False],
    # 'lr_final': [0.002, 0.001, 0.0005],
    'lr_final': [0.002, 0.001],
    'weight_decay': [0.002, 0.001, 0.0005],
    'vx_loss_d': [0.2, 0.3, 0.4, 0.5],
    'consider_old_data': [True, False],
    'inv_vx_loss_fadeout': [1., 3., 10., 30.,],
    # 'relative_kernel_lengthscale': [1/8, 1/4, 1/2],
    'relative_kernel_lengthscale': [1/8, 1/6, 1/4, 1/3, 1/2],
    'active_learning_batchsize': [128, 256, 512, 1024],
    'sweep_name': ['euler_test_random'],


    # 'nn_sobolev_weight_vx': [0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30., 100.],
    # 'inv_vx_loss_fadeout': [0.1, 0.3, 1., 3., 10., 30., 100.],

    # OUTPUT & VISUALISATION
    # (euler config here so we can keep local debugging type config in main file)
    'wandb': [True],

    'savefigs': [True],
    'wandbfigs':[False],
    'showfigs': [False],

    'ipdb_interval': [0],
}



orbits_configs = {

    # 'seed': [1,2,3,4,5,6,7,8],

    'lr_final': [0.001, 0.002, 0.005, 0.01],
    'weight_decay': [.0001, 0.0002, 0.0005, 0.001],
    # 'lr_final': [0.0001, .0002, .0005, .001, .002, .005],
    # 'weight_decay': [.0001, .0002, .0005, .001, .002, .005],

    # 'vx_loss_d': [ 0.2, 0.3, 0.4, 0.5 ],
    'nn_value_sweep': [True, False],
    'inv_vx_loss_fadeout': [3.],
    'consider_old_data': [True, False],
    'nn_layer_dim': [16, 32],
    'relative_kernel_lengthscale': [1/8, 1/4, 1/2],

    # OUTPUT & VISUALISATION
    # (euler config here so we can keep local debugging type config in main file)
    'wandb': [True],

    'savefigs': [True],
    'wandbfigs':[False],
    'showfigs': [False],

    'ipdb_interval': [0],
}

def main():
    command_list = []

    random = True

    if PROJECT_NAME == 'flatquad':
        import flatquad_landing_experiment as exp
        config_dict = flatquad_configs
        num_gpus = 1
    elif PROJECT_NAME == 'orbits':
        import orbits_experiment as exp
        config_dict = orbits_configs
        num_gpus = 0
    else:
        raise ValueError(f'Unknown project name: {PROJECT_NAME}')

    if random:
        flags_combinations = random_dict_permutations(config_dict, 512)
    else:
        flags_combinations = dict_permutations(config_dict)

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
                          num_gpus=num_gpus,
                          mode='euler',
                          duration='3:59:00',
                          prompt=True,
                          mem=16384)


if __name__ == '__main__':
    main()
