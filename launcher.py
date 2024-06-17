#!/usr/bin/env python
import sys

import numpy as np

from util import (available_gpus, dict_permutations, random_dict_permutations,
                  generate_base_command, generate_run_commands)

PROJECT_NAME = 'flatquad'
# PROJECT_NAME = 'orbits'

N_seeds = 4

# nicely working base config.
flatquad_configs_base = {

    'nn_value_sweep': [True],
    'nn_layer_dim': [128],
    'lr_final': [0.001],
    'weight_decay': [0.002],
    'vx_loss_d': [0.3],
    'consider_old_data': [True],
    'inv_vx_loss_fadeout': [1.],
    'relative_kernel_lengthscale': [0.125],
    'active_learning_batchsize': [512],
    'sweep_name': ['base'],
    'seed': list(range(N_seeds)),

    # OUTPUT & VISUALISATION
    # (euler config here so we can keep local debugging type config in main file)
    'wandb': [True],

    'savefigs': [True],
    'wandbfigs':[False],
    'showfigs': [False],

    'ipdb_interval': [0],
}

flatquad_configs_vxd = {

    'nn_value_sweep': [True],
    'nn_layer_dim': [128],
    'lr_final': [0.001],
    'weight_decay': [0.002],
    'vx_loss_d': [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'consider_old_data': [True],
    'inv_vx_loss_fadeout': [1.],
    'relative_kernel_lengthscale': [0.125],
    'active_learning_batchsize': [512],
    'sweep_name': ['vxd'],
    'seed': list(range(N_seeds)),

    # OUTPUT & VISUALISATION
    # (euler config here so we can keep local debugging type config in main file)
    'wandb': [True],

    'savefigs': [True],
    'wandbfigs':[False],
    'showfigs': [False],

    'ipdb_interval': [0],
}



flatquad_configs_rtol = {
    'nn_value_sweep': [True],
    'nn_layer_dim': [128],
    'lr_final': [0.001],
    'weight_decay': [0.002],
    'vx_loss_d': [0.3],
    'consider_old_data': [True],
    'inv_vx_loss_fadeout': [1.],
    'relative_kernel_lengthscale': [0.125],
    'active_learning_batchsize': [512],
    'sweep_name': ['rtol'],
    'seed': list(range(N_seeds)),
    'pontryagin_solver_rtol': [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4],

    # OUTPUT & VISUALISATION
    # (euler config here so we can keep local debugging type config in main file)
    'wandb': [True],

    'savefigs': [True],
    'wandbfigs':[False],
    'showfigs': [False],

    'ipdb_interval': [0],
}


flatquad_configs_dtmax = {
    'nn_value_sweep': [True],
    'nn_layer_dim': [128],
    'lr_final': [0.001],
    'weight_decay': [0.002],
    'vx_loss_d': [0.3],
    'consider_old_data': [True],
    'inv_vx_loss_fadeout': [1.],
    'relative_kernel_lengthscale': [0.125],
    'active_learning_batchsize': [512],
    'sweep_name': ['dtmax'],
    'dtmax': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
    'seed': list(range(N_seeds)),

    # OUTPUT & VISUALISATION
    # (euler config here so we can keep local debugging type config in main file)
    'wandb': [True],

    'savefigs': [True],
    'wandbfigs':[False],
    'showfigs': [False],

    'ipdb_interval': [0],
}



# sweep over weight decay
flatquad_configs_wd = {

    'nn_value_sweep': [True],
    'nn_layer_dim': [128],
    'lr_final': [0.001],
    'weight_decay': [.00001, .00002, .00005, 0.0001, 0.0002, 0.0005 , 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
    # 'weight_decay': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
    'vx_loss_d': [0.3],
    'consider_old_data': [True],
    'inv_vx_loss_fadeout': [1.],
    'relative_kernel_lengthscale': [0.125],
    'active_learning_batchsize': [512],
    'sweep_name': ['weight_decay'],
    'seed': list(range(N_seeds)),

    # OUTPUT & VISUALISATION
    # (euler config here so we can keep local debugging type config in main file)
    'wandb': [True],

    'savefigs': [True],
    'wandbfigs':[False],
    'showfigs': [False],

    'ipdb_interval': [0],
}


# sweep over batchsize
flatquad_configs_batchsize = {

    'nn_value_sweep': [True],
    'nn_layer_dim': [128],
    'lr_final': [0.001],
    'weight_decay': [0.002],
    'vx_loss_d': [0.3],
    'consider_old_data': [True],
    'inv_vx_loss_fadeout': [1.],
    'relative_kernel_lengthscale': [0.125],
    'active_learning_batchsize': [64, 128, 256, 512, 1024],
    'sweep_name': ['batchsize'],
    'seed': list(range(N_seeds)),

    # OUTPUT & VISUALISATION
    # (euler config here so we can keep local debugging type config in main file)
    'wandb': [True],

    'savefigs': [True],
    'wandbfigs':[False],
    'showfigs': [False],

    'ipdb_interval': [0],
}


# sweep over nn layer dim
flatquad_configs_layerdim = {

    'nn_value_sweep': [True],
    'nn_layer_dim': [32, 64, 128, 256, 1024],
    'lr_final': [0.001],
    'weight_decay': [0.002],
    'vx_loss_d': [0.3],
    'consider_old_data': [True],
    'inv_vx_loss_fadeout': [1.],
    'relative_kernel_lengthscale': [0.125],
    'active_learning_batchsize': [128],
    'sweep_name': ['layerdim'],
    'seed': list(range(N_seeds)),

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
    'inv_vx_loss_fadeout': [1.],
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

    random = False

    if PROJECT_NAME == 'flatquad':
        import flatquad_landing_experiment as exp
        config_dict = flatquad_configs_dtmax
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
                          mem=32768)

def flatquad_all_sweeps():

    import flatquad_landing_experiment as exp

    do_print = len(sys.argv) > 1 and sys.argv[1] in ('-p', '--print')

    command_list = []

    for config_dict in (flatquad_configs_wd, flatquad_configs_batchsize, flatquad_configs_dtmax, flatquad_configs_vxd):
        flags_combinations = dict_permutations(config_dict)
        for flags in flags_combinations:
            cmd = generate_base_command(exp, flags=flags)
            if do_print:
                print(cmd)
            command_list.append(cmd)

    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=1,
                          mode='euler',
                          duration='3:59:00',
                          prompt=True,
                          mem=16384)


if __name__ == '__main__':
    main()
    # flatquad_all_sweeps()
