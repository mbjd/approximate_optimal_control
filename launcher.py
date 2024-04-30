import flatquad_landing_experiment as exp

from util import generate_run_commands, generate_base_command, dict_permutations, available_gpus

PROJECT_NAME = 'flatquad'

flatquad_configs = {
    'T_value_target': [1., 2., 3., 5.],
    'weight_decay': [.0001, .0005, .001, .005, .01, .05, .1]
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
                          mem=16384)


if __name__ == '__main__':
    main()
