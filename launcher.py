#!/usr/bin/env python
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
    'vx_loss_d': [.001, .005, 0.01, .05, .1, .5, 1],
    'v_loss_d': [.001, .005, .01, .05, .1, .5, 1],
}

'''
flatquad_configs = {
    'T_value_target': [1.],
    'weight_decay': [.0001]
}
'''


def main():
    command_list = []
    flags_combinations = dict_permutations(flatquad_configs)

    for flags in flags_combinations:
        cmd = generate_base_command(exp, flags=flags)
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
