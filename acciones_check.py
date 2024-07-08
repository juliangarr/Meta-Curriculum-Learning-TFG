import stable_baselines3 as sb3
from Mapa import *
from ZeldaEnv import *
from Utiles import *


# Evaluar el modelo en cada uno de los mapas
model_path = f"MODELS_prueba_R/KEY/model_1.zip"
mapa = Mapa("zelda_0.txt")
env = ZeldaEnv(Task.FIND_KEY, mapa)
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
    print(f"Recompensa total: {env.state.score}")
    print(f"Pasos: {env.state.steps}\n\n")