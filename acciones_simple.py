import stable_baselines3 as sb3
from Mapa import *
from ZeldaEnv import *
from Utiles import *


# Evaluar el modelo en cada uno de los mapas
#model_path = f"models_Simple_Zelda/ZELDA_simple/model_10.zip"
model_path = f"models_Simple_Zelda/ZELDA_simple/model_2.zip"
mapa = Mapa("s_zelda_0.txt")
env = ZeldaEnv(mapa, Task.ZELDA)
env.reset()

model = sb3.A2C.load(model_path, env=env)

episodes = 3
for ep in range(episodes):
    obs, _ = env.reset()    # Extraer solo la observación
    done = False
    while not done:
        action, _ = model.predict(obs)
        print(f"Acción: {action}")
        obs, reward, done, _, _ = env.step(action)


    print(f"Episodio {ep+1} completado.")
    print(f"Recompensa total: {env.total_reward}")
    print(f"Pasos: {env.pasos}\n\n")