#!/usr/bin/env python
import flatquad_landing_experiment as exp

from util import generate_run_commands, generate_base_command, dict_permutations, available_gpus

PROJECT_NAME = 'flatquad'

flatquad_configs = {
    # 'T_value_target': [1/2, 1, 2.],
    # 'weight_decay': [.0001, .0005, .001, .005, .01, .05, .1]
    # 'seed': [1,2,3,4,5,6,7,8]
    # 'lr_init': [0.05, 0.02, 0.01, 0.005],
    # 'lr_final': [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001],

    'second_pruning_sigma': [-1., -0.5, -0.2, -0.1, 0., 0.1, 0.2, 0.5, 1., 1.5, 2., 3., 4., 5., 7., 10., 15, 20],

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
