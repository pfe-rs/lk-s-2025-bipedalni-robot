from qlearningclass import qLearning 
from twindelayedDDPGclass import TD3

algorithm=int(input("Koji algoritam zelis da pokrenes? \n 1)Q-Learning \n 2)TD3 \n"))

if(algorithm==1):
    qlearn=qLearning(n_games=2,render=True)
    qlearn.run_algorithm()
elif(algorithm==2):
    td3=TD3(n_games=2,render=True)
    td3.run_algorithm()