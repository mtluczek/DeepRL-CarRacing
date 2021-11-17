import gym
import os
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import argparse
from src.env_wrappers import EnvironmentWrappers, DiscreteCarEnvironment

WIDTH = 84
HEIGHT = 84
STACKED_FRAMES = 8
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32
TIME_STEPS = 5000000
NUM_EPISODES = 10


def train_model(model_name, buf_size, batch_size, log_path, model_path, num_steps):
    if model_name == "DQN":
        model = DQN('CnnPolicy', env, verbose=1, device='cuda', buffer_size=REPLAY_BUFFER_SIZE, batch_size=BATCH_SIZE,
                    tensorboard_log=log_path)
    else:
        if model_name == "PPO":
            model = PPO('CnnPolicy', env, verbose=1, device='cuda', tensorboard_log=log_path)
        else:
            return -1

    save_best_path = os.path.join(model_path, '/best_model')
    save_final_path = os.path.join(model_path, '/final_model')

    print(save_final_path)
    print(save_best_path)

    eval_env = model.get_env()
    eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path=save_best_path,
                                 n_eval_episodes=5,
                                 eval_freq=50000, verbose=1,
                                 deterministic=True, render=False)
    model.learn(total_timesteps=num_steps, callback=eval_callback)
    model.save(save_final_path)
    return save_final_path


def resume_train_model(model_name, log_path, model_path, environment, num_steps):
    load_path = os.path.join(model_path, model_name, "best_model")
    best_path = os.path.join(model_path, model_name, "best_model")
    save_final_path = os.path.join(model_path, model_name, "final_model")
    print(f"Logging to Tensorboard: {log_path}")
    print(f"Saving models to: {model_path}")

    if model_name == "DQN":
        model = DQN.load(best_path, tensorboard_log=log_path)
        print(f"Resuming DQN training")
    else:
        if model_name == "PPO":
            model = PPO.load(best_path, tensorboard_log=log_path)
            print(f"Resuming PPO training")
        else:
            return -1

    model.set_env(environment)
    eval_callback = EvalCallback(eval_env=model.get_env(), best_model_save_path=best_path,
                                 n_eval_episodes=5,
                                 eval_freq=50000, verbose=1,
                                 deterministic=True, render=False)
    model.learn(total_timesteps=num_steps, callback=eval_callback, reset_num_timesteps=False)
    model.save(save_final_path)
    return save_final_path


def test_model(environment, model_name, model_path, episodes):
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=episodes, render=True)
    if model_name == "DQN":
        model = DQN.load(model_path, environment)
    else:
        if model_name == "PPO":
            model = PPO.load(model_path, environment)
        else:
            return -1
    environment = gym.wrappers.Monitor(environment, directory="monitor", force=True)
    model.set_env(environment)
    environment = model.get_env()

    for episode in range(1, episodes + 1):
        obs = environment.reset()
        done = False
        score = 0
        while not done:
            environment.render(mode='rgb_array')
            action, _states = model.predict(obs)
            obs, reward, done, info = environment.step(action)
            score += reward
        print("Episode:{} Score:{}".format(episode, score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="test", help="Choose mode: train/test/resume")
    parser.add_argument("-a", "--agent", default="DQN", help="Reinforcement learning model/agent to use")
    parser.add_argument("-l", "--log_path", default=os.path.join(".", "Training", "Logs"),
                        help="Directory to store logs")
    parser.add_argument("-s", "--save_model_path", default=os.path.join(".", "Training", "Saved_Models"),
                        help="Directory to store models")
    parser.add_argument("-b", "--buffer_size", default=REPLAY_BUFFER_SIZE, help="Replay buffer size for DQN")
    parser.add_argument("-t", "--time_steps", default=TIME_STEPS, help="Number of training time steps")
    parser.add_argument("-m", "--model_path", help="Provide path to model file")
    args = parser.parse_args()

    if args.mode == "test" and (args.model_path is None):
        parser.error("--mode=test requires providing valid --model_path")

    if args.agent == "PPO":
        env = gym.make("CarRacing-v0")
    else:
        env = DiscreteCarEnvironment(gym.make("CarRacing-v0"))

    env_wrappers = EnvironmentWrappers(WIDTH, HEIGHT, STACKED_FRAMES)
    funcs = [env_wrappers.resize, env_wrappers.grayscale, env_wrappers.frame_stack]
    env = env_wrappers.observation_wrapper(env, funcs)
    print(env.observation_space.shape)

    if args.mode == "train":
        saved_path = train_model(args.agent, args.buffer_size, BATCH_SIZE,
                                 args.log_path, args.save_model_path, args.time_steps)
        print(f'Model successfully trained after {args.time_steps} time steps. Results saved in model '
              f'file {saved_path}')
    else:
        if args.mode == "resume":
            saved_path = resume_train_model(args.agent, args.log_path, args.save_model_path, env, args.time_steps)
            print(f'Model successfully trained after {args.time_steps} time steps. Results saved in model '
                  f'file {saved_path}')
        if args.mode == "test":
            test_model(env, args.agent, args.model_path, NUM_EPISODES)
        else:
            exit(-1)

    env.close()
