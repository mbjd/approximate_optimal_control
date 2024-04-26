import argparse
import datetime
from datetime import datetime

import imageio
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import wandb
from brax import envs
from jax.nn import swish
from mbpo.optimizers.policy_optimizers.sac.sac_brax_env import SAC

ENTITY = 'dbalduin'


def experiment(env_name: str = 'inverted_pendulum',
               backend: str = 'generalized',
               project_name: str = 'GPUSpeedTest',
               num_timesteps: int = 1_000_000):
    assert env_name in ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum',
                        'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    assert backend in ['generalized', 'positional', 'spring']
    env = envs.get_environment(env_name=env_name,
                               backend=backend)

    action_repeat = 1
    episode_length = 1000
    discount_factor = 0.99

    config = dict(env_name=env_name,
                  backend=backend)

    wandb.init(
        project=project_name,
        dir='/cluster/scratch/' + ENTITY,
        config=config,
    )

    optimizer = SAC(
        environment=env,
        num_timesteps=num_timesteps,
        episode_length=episode_length,
        action_repeat=action_repeat,
        num_env_steps_between_updates=20,
        num_envs=32,
        num_eval_envs=4,
        lr_alpha=3e-4,
        lr_policy=3e-4,
        lr_q=3e-4,
        wd_alpha=0.,
        wd_policy=0.,
        wd_q=0.,
        max_grad_norm=1e5,
        discounting=discount_factor,
        batch_size=32,
        num_evals=20,
        normalize_observations=True,
        reward_scaling=1.,
        tau=0.005,
        min_replay_size=10 ** 4,
        max_replay_size=10 ** 5,
        grad_updates_per_step=20 * 32,  # should be num_envs * num_env_steps_between_updates
        deterministic_eval=True,
        init_log_alpha=0.,
        policy_hidden_layer_sizes=(32,) * 5,
        policy_activation=swish,
        critic_hidden_layer_sizes=(128,) * 4,
        critic_activation=swish,
        wandb_logging=True,
        return_best_model=True,
    )

    xdata, ydata = [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics['eval/episode_reward'])
        # plt.xlim([0, train_fn.keywords['num_timesteps']])
        # plt.ylim([min_y, max_y])
        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.plot(xdata, ydata)
        plt.show()

    print('Before inference')
    policy_params, metrics = optimizer.run_training(key=jr.PRNGKey(0), progress_fn=progress)
    print('After inference')

    # Now we plot the evolution
    pseudo_policy = optimizer.make_policy(policy_params, deterministic=True)

    def policy(obs):
        return pseudo_policy(obs, key_sample=jr.PRNGKey(0))

    @jax.jit
    def step(state, _):
        u = policy(state.obs)[0]
        print('Step')
        print(f'Time to go {u[-1]}')
        next_state = env.step(state, u)
        return next_state

    time_to_jit = times[1] - times[0]
    time_to_train = times[-1] - times[1]

    print(f'time to jit: {time_to_jit}')
    print(f'time to train: {time_to_train}')

    state = env.reset(rng=jr.PRNGKey(0))
    trajectory = []
    trajectory.append(state)
    rewards = []
    for i in range(episode_length):
        state = step(state, None)
        trajectory.append(state)
        rewards.append(state.reward)

    total_reward = sum(rewards)
    wandb.log({'total_reward': total_reward})

    # video_frames = env.render([s.pipeline_state for s in trajectory], camera='track')
    video_frames = env.render([s.pipeline_state for s in trajectory])

    with imageio.get_writer('video.mp4', fps=int(1 / env.dt)) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    wandb.log({"video": wandb.Video("video.mp4", fps=int(1 / env.dt), format="gif")})
    wandb.finish()


def main(args):
    experiment(env_name=args.env_name,
               backend=args.backend,
               project_name=args.project_name,
               num_timesteps=args.num_timesteps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='inverted_pendulum')
    parser.add_argument('--backend', type=str, default='generalized')
    parser.add_argument('--project_name', type=str, default='GPUSpeedTest')
    parser.add_argument('--num_timesteps', type=int, default=40_000)

    args = parser.parse_args()
    main(args)
