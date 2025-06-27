import gymnasium as gym
import numpy as np
import math
import random
from collections import defaultdict
import wandb

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

class qLearning():
    def __init__(self,n_games=100,alpha=0.01,gamma=0.99,render=False):
        if render:
            self.env=gym.make("BipedalWalker-v3", render_mode="human") 
        else:
            self.env=gym.make("BipedalWalker-v3") 
        self.qTable=defaultdict( lambda: np.zeros((10, 10, 10, 10)))
        self.n_games=n_games
        self.alpha=alpha
        self.gamma=gamma
        self.epsilon=1
        wandb.init(
        # Set the project where this run will be logged
        project="bipedal-Ql",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"Q-learning",
        # Track hyperparameters and run metadata
        config={
        "algorithm": "Q-Learning",
        "alpha": self.alpha,
        "gamma": self.gamma,
        "episodes": self.n_games,
        })
        self.highscore=-100000
        self.observation,info=self.env.reset(seed=42)
        self.cumulated_reward=0
    def updateQTable(self,state, action, reward, nextState = None):
        current = self.qTable[state][action] #trenutni akcija-stanje par

        if nextState is not None:
            qValues = self.qTable[nextState] # Uzimamo qValues svih akcija za sledece stanje
            qNext = np.max(qValues) # Uzimamo najvecu qValue - zbog pretpostavke da cemo dobiti najvecu nagradu
        else:
            qNext = 0 # Ako je proces terminated ili trunctuted, nece biti nextState, samim tim nece biti ni nagrade

        target = reward + (self.gamma * qNext) #Bellmanovom formulom racunamo target qValue

        new_value = current + (self.alpha * (target - current)) #Temporal Difference Update (TD Update) - formula za racunanje novih qValues
        return new_value
    def convertNextAction(nextAction): #pretvaramo diskretne podatke u kontinualne da bi BipedalWalker mogao da hoda
        action = []

        for i in range(len(nextAction)):
            nextVal = nextAction[i]/9*2-1   # /9 da bi prebacio trenutne vrednosti u range [0,1]
                                            # *2 da bismo imali vrednosti [0,2]
                                            # -1 da bismo imali vrednosti [-1,1] sto nam je uslov zadatka
            action.append(nextVal)
        return tuple(action)

    def getNextAction(self,state):
        if random.random() < self.epsilon:
            action = ()
            for i in range(0,4):
                action += (random.randint(0,9),) #daj random sile na zglobove
        else:
            action = np.unravel_index(np.argmax(self.qTable[state]), self.qTable[state].shape) #trazi najvecu vrednost u qTable
            #prvo mora da ga konvertuje u 1D da bi nasao max, onda ga vraca u niz [10,10,10,10]
        return action
    def discretizeState(self,state): #prebacuje kontinualne podatke u diskretne kako ne bismo imali beskonacno mnogo podataka u qTable
        discreteState = []
        for i in range (len(state)): #prolazimo kroz svih 14 state value-a
            print()
            normalized = (state[i] - stateBounds[i][0]) / (stateBounds[i][1] - stateBounds[i][0]) #pretvaramo sve u range [0,1]
            scaled = normalized*19 #skaliramo jer koristimo 20 diskretnih binova
            index = int(scaled) 
            discreteState.append(index)
        return tuple(discreteState)
    def runOneEpisode(self, i):

        self.env.render()
        print("Episode: " + str(i))

        obs, info = self.env.reset() #treba nam samo obzervacija, info zanemarujemo za treniranje
        state = self.discretizeState(obs[:14])

        self.cumulated_reward = 0
        self.epsilon = float(1.0 / (i*0.004))

        while True:
            nextActionDisc = self.getNextAction(state)
            nextAction = self.convertNextAction(nextActionDisc)

            nextState, reward, terminated, truncated, info = self.env.step(nextAction)
            done = terminated or truncated

            nextState = self.discretizeState(nextState[0:14])

            cumulated_reward += reward

            self.qTable[state][nextActionDisc] = self.updateQTable(state,nextActionDisc,reward,nextState)
            state = nextState

            if done:
                break
        
        if cumulated_reward > self.highscore:
            self.highscore = cumulated_reward
        
        return cumulated_reward
    def run_algorithm(self):
        for i in range(1, self.n_games+1):
            epScore = self.runOneEpisode(i)
            print("Zavrsena epizoda. Nacrtana tacka na grafiku.")
            wandb.log({"episode": i, "epsilon":self.epsilon,"score": epScore})
        print("Zavrseno treniranje. HIGHSCORE: " + str(HIGHSCORE))
        wandb.finish()
        self.env.close()