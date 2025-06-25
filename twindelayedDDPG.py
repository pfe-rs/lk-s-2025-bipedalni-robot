import gymnasium as gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

class ReplayBuffer(): #cuva prethodno nauceno
    def __init__(self, max_size, input_shape, n_actions):
        #max_size - max broj tranzicija sto cuva
        #input_shape - oblik obzervovanih stanja (24,) u nasem slucaju
        #n_actions - broj akcija, ali s obzirom da radimo sa kontinualnim sistemom
        #            predstavlja broj dimenzija akcije 
        self.mem_size = max_size
        self.mem_counter = 0 #sluzi da brojimo koliko smo tranzicija sacuvali
        self.state_memory = np.zeros((self.mem_size, *input_shape)) #cuva trenutno stanje (input_shape)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape)) #cuva sledece stanje
        self.action_memory = np.zeros((self.mem_size, n_actions)) #cuva akciju koja je preduzeta
        self.reward_memory = np.zeros(self.mem_size) #cuva dobijenu nagradu kao skalar 
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool ) #cuva da li je epizoda zavrsena
    
    def store_transitions(self, state, action, reward, new_state, done): #metoda za dodavanje jednog novog iskustva
        index = self.mem_counter % self.mem_size #racunamo index u nizu gde stavljamo novo iskustvo
                                                 #ako buffer nije pun koristi counter
                                                 #ako je pun, vraca se na pocetak i overwrite-uje postojece
                                                 # % mem_size osigurava da smo u okviru [0,max_size-1]
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_counter += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size) #ne zelimo da treniramo agenta na nulama
                                                       #ako bismo uzimali samo mem_counter izasli bismo iz okvira niza
                                                       #a mem_size na pocetku ima nule u sebi 
        batch = np.random.choice(max_mem, batch_size) #biramo random indexse iz max_mem u velicini batch-a
        
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones #vraca ceo batch
    
class CriticNetwork(nn.Module): #evaluira koliko je neka akcija dobra za neko stanje, dalje Q(s,a)
    def __init__(self, beta, input_dims, fullyConnectedL1_dims, fullyConnectedL2_dims, n_actions, name, checkpoint_dir='./model'):
        #beta - learning rate, input_dims - dimenzije stanja (24) 
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fullyConnectedL1_dims = fullyConnectedL1_dims
        self.fullyConnectedL2_dims = fullyConnectedL2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name+'_td3')

        self.fullyConnectedL1 = nn.Linear(self.input_dims[0]+n_actions, self.fullyConnectedL1_dims)#input layer
        self.fullyConnectedL2 = nn.Linear(self.fullyConnectedL1_dims, self.fullyConnectedL2_dims)#hiddem layer
        self.q1 = nn.Linear(self.fullyConnectedL2_dims, 1)#output layer, daje prediktovanu Q vrednost (skalar)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)
    
    def forward(self, state, action): #unapred prolazak kroz mrezu
        q1_action_value = self.fullyConnectedL1(T.cat([state,action],dim=1)) #moramo da spojimo akciju i stanje jer layer to zahteva
        q1_action_value = F.relu(q1_action_value) #aktivacija ReLU f-jom koja dodaje nelinearnost
        q1_action_value = self.fullyConnectedL2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)#output layer sa Q vrednoscu

        return q1
    
    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fullyConnectedL1_dims, fullyConnectedL2_dims, n_actions, name, checkpoint_dir='./model'):
        super(ActorNetwork,self).__init__()
        self.input_dims = input_dims
        self.fullyConnectedL1_dims = fullyConnectedL1_dims
        self.fullyConnectedL2_dims = fullyConnectedL2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name+'_td3')

        self.fullyConnectedL1 = nn.Linear(self.input_dims[0], self.fullyConnectedL1_dims)
        self.fullyConnectedL2 = nn.Linear(self.fullyConnectedL1_dims, self.fullyConnectedL2_dims)
        self.mu = nn.Linear(self.fullyConnectedL2_dims, self.n_actions)#zove se mu jer je po konvenciji prihvacen naziv za deterministicke 

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state): #koristimo feedforward NN
        prob = self.fullyConnectedL1(state)
        prob = F.relu(prob)
        prob = self.fullyConnectedL2(prob)
        prob = F.relu(prob)

        mu = T.tanh(self.mu(prob))#tanh prebacuje output da bude u okviru [-1,1]

        return mu
    
    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, update_actor_interval=2,
                 warmup=1000, n_actions=2, max_size=1000000, layer1_size=400, layer2_size=300, batch_size=100, noise =0.1):
                #vrednosti preuzete iz referentnog rada
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_interval = update_actor_interval

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='actor')
        self.critic1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions,name='critic1')
        self.critic2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions,name='critic2')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='target_actor')
        self.target_critic1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions,name='target_critic1')
        self.target_critic2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions,name='target_critic2')

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step<self.warmup:#ako nije izasao iz warmup-a, uzima random parametre sa sumom da bi explorovao
            mu = T.tensor(np.random.normal(scale=self.noise,size=(self.n_actions,)))
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu=self.actor.forward(state).to(self.actor.device)#inace feedforwarduj kroz actor-a
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise),dtype=T.float).to(self.actor.device)#dodaj noise
        mu_prime = T.clamp(mu_prime, self.min_action[0],self.max_action[0])#clippuj noise da ne prelazi ocekivane granice
        self.time_step+=1

        return mu_prime.cpu().detach().numpy()
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transitions(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_counter < self.batch_size: #ne treniramo model ako nema dovoljno podataka za jedan batch
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic1.device)
        done = T.tensor(done).to(self.critic1.device)
        new_state = T.tensor(new_state,dtype=T.float).to(self.critic1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic1.device)

        #pravimo target mreze
        target_actions = self.target_actor.forward(new_state)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2, size=target_actions.shape), dtype=T.float).to(self.actor.device), -0.5, 0.5)
        target_actions = T.clamp(target_actions,self.min_action[0],self.max_action[0])

        #pravimo qValues za sledecu
        new_q1 = self.target_critic1.forward(new_state, target_actions)
        new_q2 = self.target_critic2.forward(new_state,target_actions)
        
        #pravimo qValues za trenutnu
        q1 = self.critic1.forward(state, action)
        q2 = self.critic2.forward(state, action)

        #ne ocekujemo nagradu za terminalna/krajnja stanja
        new_q1[done] = 0.0
        new_q2[done] = 0.0

        new_q1 = new_q1.view(-1)
        new_q2 = new_q2.view(-1)
        new_critic_value = T.min(new_q1, new_q2)#Double Q-learning, biramo manji qValue

        target = reward + self.gamma * new_critic_value #Bellmanova jednacina
        target = target.view(self.batch_size, 1)

        #Update-ujemo critic-a racunanjem loss f-ja i unazad propagiranjem da bi ih apply-ovali
        self.critic1.optimizer.zero_grad() #clear-ujemo prethodno sacuvane gradijente
        self.critic2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target,q1) #racunamo mean squared loss
        q2_loss = F.mse_loss(target,q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        #Delayed update-ovanje actor-a
        self.learn_step_counter += 1

        if self.learn_step_counter % self.update_actor_interval != 0: #svaki drugi put
            return
        
        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:#tau je onaj vrlo mali broj
            tau = self.tau
        
        actor_params = self.actor.named_parameters()
        critic1_params = self.critic1.named_parameters()
        critic2_params = self.critic2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic1_params = self.target_critic1.named_parameters()
        target_critic2_params = self.target_critic2.named_parameters()

        critic1 = dict(critic1_params)
        critic2 = dict(critic2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic1 = dict(target_critic1_params)
        target_critic2 = dict(target_critic2_params)

        #soft update-ovanje svega po formuli
        for name in critic1:
            critic1[name] = tau*critic1[name].clone() + (1-tau)*target_critic1[name].clone()
        for name in critic2:
            critic2[name] = tau*critic2[name].clone() + (1-tau)*target_critic2[name].clone()
        for name in actor:
            actor[name] = tau*actor[name].clone() + (1-tau)*target_actor[name].clone()

        self.target_critic1.load_state_dict(critic1)
        self.target_critic2.load_state_dict(critic2)
        self.target_actor.load_state_dict(actor)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.target_critic1.save_checkpoint()
        self.target_critic2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()
        self.target_critic1.load_checkpoint()
        self.target_critic2.load_checkpoint()

env = gym.make('BipedalWalker-v3', render_mode='human')
agent = Agent(alpha=0.001, beta=0.001,
        input_dims=env.observation_space.shape, tau=0.005,
        env=env, batch_size=100, layer1_size=400, layer2_size=300,
        n_actions=env.action_space.shape[0])
n_games = 1000
filename = 'plots/' + 'walker_' + str(n_games) + '_games.png'

best_score = -1000
score_history = []

agent.load_models()

for i in range(n_games):
    observation, info = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.remember(observation, action, reward, observation_, done)
        agent.learn()
        score += reward
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    print('episode ', i, 'score %.1f' % score,
            'average score %.1f' % avg_score)

x = [i+1 for i in range(n_games)]
plot_learning_curve(x, score_history, filename)