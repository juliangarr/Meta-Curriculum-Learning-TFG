import stable_baselines3 as sb3
from ZeldaEnv import *
import os
import csv


def evaluar_tarea(model_path, csv_filename, csv_dir, extension, level_files, posiciones, Task):
    # Crear los mapas
    mapas = [Mapa(f"{file}") for file in level_files]
    
    csv_path = os.path.join(csv_dir, f"{csv_filename}.csv")

    reward_mean = [0.0, 0.0, 0.0, 0.0, 0.0]
    score_mean = [0.0, 0.0, 0.0, 0.0, 0.0]
    wins = [0.0, 0.0, 0.0, 0.0, 0.0]
    step_mean = [0.0, 0.0, 0.0, 0.0, 0.0]

    # Crear el archivo CSV
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow([f"TAREA: {Task.name}_{extension}"])
        writer.writerow([])
        writer.writerow([])


        for mapa, i, pos in zip(mapas, range(len(mapas)), posiciones):
            env = ZeldaEnv(Task, mapa, pos_jugador=pos)
            env.reset()

            model = sb3.A2C.load(model_path, env=env)

            episodes = 20
            writer.writerow([f"Mapa {i}: {level_files[i]}"])
            writer.writerow(["Episodio", "TotalReward", "Pasos", "Score"])

            for ep in range(episodes):
                obs, _ = env.reset()    # Extraer solo la observación
                done = False
                while not done:
                    action, _ = model.predict(obs)
                    obs, reward, done, _, _ = env.step(action)

                #print(f"Episodio {ep+1} completado por tarea {i}.")
                #print(f"Recompensa total: {env.total_reward}")
                writer.writerow([ep+1, env.total_reward, env.state.steps, env.state.score])
                reward_mean[i] += env.total_reward
                score_mean[i] += env.state.score
                step_mean[i] += env.state.steps
                if env.state.score == 1000:
                    wins[i] += 1

            
            writer.writerow(["MEAN", reward_mean[i]/episodes, step_mean[i]/episodes, score_mean[i]/episodes])
            writer.writerow(["WINS", wins[i]])
            
            writer.writerow([])
            writer.writerow([])

        print(reward_mean)
        print(score_mean)
        print(step_mean)

        reward_mean = np.array(reward_mean)/episodes
        score_mean = np.array(score_mean)/episodes
        step_mean = np.array(step_mean)/episodes

        print(reward_mean)
        print(score_mean)
        print(step_mean)
        
        r_mean = sum(reward_mean) / len(reward_mean)
        s_mean = sum(score_mean) / len(score_mean)
        st_mean = sum(step_mean) / len(step_mean)
        w = sum(wins)
        
        writer.writerow(["", "reward_M", "step_M", "score_M"])
        writer.writerow(["TOTAS_MEANS", r_mean, st_mean, s_mean])
        writer.writerow(["TOTAL_WINS", w])
        writer.writerow(["Win_MEAN", w/len(wins)])
