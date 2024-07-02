import stable_baselines3 as sb3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from Mapa import *
from ZeldaEnv import *
from Utiles import *

# Crear el entorno
mapa = Mapa("find_key_0.txt")
env = ZeldaEnv(mapa, Task.FIND_KEY)  # Puedes cambiar la tarea a Task.FIND_DOOR o Task.KILL_ENEMIES
#env = Monitor(env,  filename='./logs')  # Monitorizar el entorno para obtener estadísticas

# Crear el modelo PPO
#model = sb3.PPO('MlpPolicy', env, verbose=1, tensorboard_log=logs_dir, learning_rate=0.0001, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1, target_kl=None, tensorboard_log=None, create_eval_env=False, policy_kwargs=None, verbose=0, seed=None, device='auto', _init_setup_model=True)
model = sb3.PPO('MlpPolicy', env, verbose=1)

TIMESTEPS = 10000

model.learn(total_timesteps=TIMESTEPS)