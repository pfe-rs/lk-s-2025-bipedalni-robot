import gymnasium as gym
import numpy as np
import math
import random
from collections import defaultdict
import wandb
import glob
from gymnasium.wrappers import RecordVideo
import os

run=6
NUM_OF_EPISODES = 1000 #broj epizoda koliko treniramo model
ALPHA = 0.01 #learning rate
GAMMA = 0.99 #discount factor
HIGHSCORE = -200 #treba da bude manji od najmanje vrednosti koju mozemo dobiti

stateBounds = [(0, math.pi),   # Ugao tela robota
               (-2, 2),         # Ugaona brzina tela robota
               (-1, 1),         # Brzina x
               (-1, 1),         # Brzina y
               (0, math.pi),    # Ugao prve noge
               (-2, 2),         # Ugaona brzina prvog zgloba
               (0, math.pi),    # Ugao prvog zgloba
               (-2, 2),         # Ugaona brzina drugog zgloba
               (0, 1),          # Sila prve noge
               (0, math.pi),    # Ugao treceg zgloba
               (-2, 2),         # Ugaona brzina treceg zgloba
               (0, math.pi),    # Ugao cetvrtog zgloba
               (-2, 2),         # Ugaona brzina cetvrtog zgloba
               (0, 1)]          # Sila druge noge

actionBounds = (-1, 1)

def discretizeState(state): #prebacuje kontinualne podatke u diskretne kako ne bismo imali beskonacno mnogo podataka u qTable
    discreteState = []
    for i in range (len(state)): #prolazimo kroz svih 14 state value-a
        normalized = (state[i] - stateBounds[i][0]) / (stateBounds[i][1] - stateBounds[i][0]) #pretvaramo sve u range [0,1]
        scaled = normalized*19 #skaliramo jer koristimo 20 diskretnih binova
        index = int(scaled) 
        discreteState.append(index)
    return tuple(discreteState)

def getNextAction(qTable, state, epsilon):
    if random.random() < epsilon:
        action = ()
        for i in range(0,4):
            action += (random.randint(0,9),) #daj random sile na zglobove
    else:
        action = np.unravel_index(np.argmax(qTable[state]), qTable[state].shape) #trazi najvecu vrednost u qTable
        #prvo mora da ga konvertuje u 1D da bi nasao max, onda ga vraca u niz [10,10,10,10]
    return action

def convertNextAction(nextAction): #pretvaramo diskretne podatke u kontinualne da bi BipedalWalker mogao da hoda
    action = []

    for i in range(len(nextAction)):
        nextVal = nextAction[i]/9*2-1   # /9 da bi prebacio trenutne vrednosti u range [0,1]
                                        # *2 da bismo imali vrednosti [0,2]
                                        # -1 da bismo imali vrednosti [-1,1] sto nam je uslov zadatka
        action.append(nextVal)
    return tuple(action)

def updateQTable(qTable, state, action, reward, nextState = None):
    global ALPHA
    global GAMMA

    current = qTable[state][action] #trenutni akcija-stanje par

    if nextState is not None:
        qValues = qTable[nextState] # Uzimamo qValues svih akcija za sledece stanje
        qNext = np.max(qValues) # Uzimamo najvecu qValue - zbog pretpostavke da cemo dobiti najvecu nagradu
    else:
        qNext = 0 # Ako je proces terminated ili trunctuted, nece biti nextState, samim tim nece biti ni nagrade

    target = reward + (GAMMA * qNext) #Bellmanovom formulom racunamo target qValue

    new_value = current + (ALPHA * (target - current)) #Temporal Difference Update (TD Update) - formula za racunanje novih qValues
    return new_value

def runOneEpisode(env, i, qTable):
    global HIGHSCORE

    env.render()
    print("Episode: " + str(i))

    obs, info = env.reset() #treba nam samo obzervacija, info zanemarujemo za treniranje
    state = discretizeState(obs[:14])

    cumulated_reward = 0
    epsilon = float(1.0 / (i*0.004))

    while True:
        nextActionDisc = getNextAction(qTable,state,epsilon)
        nextAction = convertNextAction(nextActionDisc)

        nextState, reward, terminated, truncated, info = env.step(nextAction)
        done = terminated or truncated

        nextState = discretizeState(nextState[0:14])

        cumulated_reward += reward

        qTable[state][nextActionDisc] = updateQTable(qTable,state,nextActionDisc,reward,nextState)
        state = nextState

        if done:
            break
    
    if cumulated_reward > HIGHSCORE:
        HIGHSCORE = cumulated_reward
    
    return cumulated_reward,epsilon
    


env = gym.make("BipedalWalker-v3", render_mode="rgb_array") # Inicijalizujemo enviroment
save_ep=125
env = RecordVideo(env,
                            video_folder="q-learning-training",
                            name_prefix="training",
                            episode_trigger=lambda x: x % save_ep== 0  # Only record every 250th episode
                            )
qTable = defaultdict( lambda: np.zeros((10, 10, 10, 10))) #pravimo qTable kao 4D niz

wandb.init(
      # Set the project where this run will be logged
      project="bipedal-Ql",
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"experiment_{run}",
      # Track hyperparameters and run metadata
      config={
      "algorithm": "Q-Learning",
      "alpha": ALPHA,
      "gamma": GAMMA,
      "episodes": NUM_OF_EPISODES,
      })


# resetujemo environment kako bismo mogli generisati prvu epizodu
observation, info = env.reset(seed=42)

for i in range(1, NUM_OF_EPISODES+1):
    epScore,epsilon = runOneEpisode(env, i, qTable)
    print("Zavrsena epizoda. Nacrtana tacka na grafiku.")
    wandb.log({"episode": i, "epsilon":epsilon,"score": epScore})

print("Zavrseno treniranje. HIGHSCORE: " + str(HIGHSCORE))
video_folder = "/home/pfe-admin/Desktop/bipedal_projekat/lk-s-2025-bipedalni-robot/q-learning-training"
video_files = glob.glob(os.path.join(video_folder, "*.mp4"))

for video_path in video_files:
    video_name = os.path.basename(video_path)
    wandb.log({
        video_name: wandb.Video(video_path, format="mp4")
    })

wandb.finish()
env.close()