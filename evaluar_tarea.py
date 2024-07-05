import stable_baselines3 as sb3
from ZeldaEnv import *
import os
import csv


def evaluar_tarea(model_path, csv_filename, csv_dir, extension, level_files, posiciones, Task):
    # Crear los mapas
    mapas = [Mapa(f"{file}") for file in level_files]
    
    csv_path = os.path.join(csv_dir, f"{csv_filename}.csv")

    # Crear el archivo CSV
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow([f"TAREA: {Task.name}_{extension}"])
        writer.writerow([])
        writer.writerow([])


        for mapa, i, pos in zip(mapas, range(len(mapas)), posiciones):
            env = ZeldaEnv(mapa, Task, pos_jugador=pos)
            env.reset()

            model = sb3.A2C.load(model_path, env=env)

            episodes = 10
            writer.writerow([f"Mapa {i}: {level_files[i]}"])
            writer.writerow(["Episodio", "TotalReward", "Pasos"])

            for ep in range(episodes):
                obs, _ = env.reset()    # Extraer solo la observación
                done = False
                while not done:
                    action, _ = model.predict(obs)
                    obs, reward, done, _, _ = env.step(action)

                #print(f"Episodio {ep+1} completado por tarea {i}.")
                #print(f"Recompensa total: {env.total_reward}")
                writer.writerow([ep+1, env.total_reward, env.pasos])
            
            writer.writerow([])
