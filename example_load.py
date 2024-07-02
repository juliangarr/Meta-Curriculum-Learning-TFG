import stable_baselines3 as sb3
from Mapa import *
from ZeldaEnv import *
from Utiles import *

models_dir  = "models/PPO"

# Crear el entorno
mapa = Mapa("find_key_0.txt")
env = ZeldaEnv(mapa, Task.FIND_KEY)  # Puedes cambiar la tarea a Task.FIND_DOOR o Task.KILL_ENEMIES
env.reset()

model_path = f"{models_dir}/model_29.zip"

model = sb3.PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

    print(f"Episodio {ep+1} completado.")

    
env.close()

