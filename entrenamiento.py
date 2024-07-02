import stable_baselines3 as sb3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from Mapa import *
from ZeldaEnv import *
from Utiles import *

import os

models_dir  = "models/PPO"
logs_dir    = "logs/PPO"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Crear el entorno
mapa = Mapa("find_key_0.txt")
env = ZeldaEnv(mapa, Task.FIND_KEY)  # Puedes cambiar la tarea a Task.FIND_DOOR o Task.KILL_ENEMIES
#env = Monitor(env,  filename='./logs')  # Monitorizar el entorno para obtener estadísticas

# Crear el modelo PPO
#model = sb3.PPO('MlpPolicy', env, verbose=1, tensorboard_log=logs_dir, learning_rate=0.0001, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1, target_kl=None, tensorboard_log=None, create_eval_env=False, policy_kwargs=None, verbose=0, seed=None, device='auto', _init_setup_model=True)
model = sb3.PPO('MlpPolicy', env, verbose=1, tensorboard_log=logs_dir)

TIMESTEPS = 10000
for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/model_{i}")


'''
episodes = 10

for _ in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        #action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(env.action_space.sample())
        #env.render()
        #print(f"Action: {action}, Reward: {reward}, Done: {done}")

env.close()

# Evaluar el modelo
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
'''