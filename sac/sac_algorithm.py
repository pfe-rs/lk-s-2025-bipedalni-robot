import gymnasium as gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.distributions.normal import Normal
import wandb
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import glob

class ReplayBuffer(): #isti kao i za TD3
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_counter = 0 
        self.state_memory = np.zeros((self.mem_size, *input_shape)) 
        self.new_state_memory = np.zeros((self.mem_size, *input_shape)) 
        self.action_memory = np.zeros((self.mem_size, n_actions)) 
        self.reward_memory = np.zeros(self.mem_size) 
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool )
    
    def store_transitions(self, state, action, reward, new_state, done): #metoda za dodavanje jednog novog iskustva
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_counter += 1
    
    def sample_buffer(self, batch_size): #biramo mini-batch
        max_mem = min(self.mem_counter, self.mem_size) 
        batch = np.random.choice(max_mem, batch_size) 
        
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones 
    
class CriticNetwork(nn.Module): #evaluira koliko je neka akcija dobra za neko stanje, dalje Q(s,a); isti kao za TD3
    def __init__(self, beta, input_dims, fullyConnectedL1_dims, fullyConnectedL2_dims, n_actions, name, checkpoint_dir='./model'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fullyConnectedL1_dims = fullyConnectedL1_dims
        self.fullyConnectedL2_dims = fullyConnectedL2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name+'_sac')

        self.fullyConnectedL1 = nn.Linear(self.input_dims[0]+n_actions, self.fullyConnectedL1_dims)
        self.fullyConnectedL2 = nn.Linear(self.fullyConnectedL1_dims, self.fullyConnectedL2_dims)
        self.q1 = nn.Linear(self.fullyConnectedL2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)
    
    def forward(self, state, action): 
        q1_action_value = self.fullyConnectedL1(T.cat([state,action],dim=1))
        q1_action_value = F.relu(q1_action_value) 
        q1_action_value = self.fullyConnectedL2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1
    
    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module): #predvidja očekivanu Q vrednost po trenutnom policy-ju -> stabilizuje učenje 
    def __init__(self, beta, input_dims, fullyConnectedL1_dims=256, fullyConnectedL2_dims=256, name="value", checkpoint_dir="./model"):
        super(ValueNetwork, self).__init__()
        self.beta = beta
        self.input_dims = input_dims
        self.fullyConnectedL1_dims = fullyConnectedL1_dims
        self.fullyConnectedL2_dims = fullyConnectedL2_dims 
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, name+"_sac")

        self.fullyConnectedL1 = nn.Linear(*self.input_dims, self.fullyConnectedL1_dims)
        self.fullyConnectedL2 = nn.Linear(self.fullyConnectedL1_dims, self.fullyConnectedL2_dims)
        self.v = nn.Linear(self.fullyConnectedL2_dims,1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)
    
    def forward(self, state):
        state_value = self.fullyConnectedL1(state)
        state_value = F.relu(state_value)
        state_value = self.fullyConnectedL2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v 
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fullyConnectedL1_dims=256, fullyConnectedL2_dims=256, n_actions=2, name='actor', checkpoint_dir="./model"):
        super(ActorNetwork, self).__init__()
        self.alpha = alpha
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fullyConnectedL1_dims = fullyConnectedL1_dims
        self.fullyConnectedL2_dims = fullyConnectedL2_dims
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.reparameterization_noise = 1e-6 #vrednost preuzeta iz referentnog rada

        self.fullyConnectedL1 = nn.Linear(*self.input_dims, self.fullyConnectedL1_dims)
        self.fullyConnectedL2 = nn.Linear(self.fullyConnectedL1_dims, self.fullyConnectedL2_dims)
        self.mu = nn.Linear(self.fullyConnectedL2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fullyConnectedL2_dims, self.n_actions)#standardna devijacija

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device) 

    def forward(self, state):
        prob = self.fullyConnectedL1(state)
        prob = F.relu(prob)
        prob = self.fullyConnectedL2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparameterization_noise, max = 1)

        return mu, sigma
    
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)#pravi gausovu rasporedlu gde je centar dimenyija akcije, a širi se za standardnu devijaciju

        if reparameterize:
            actions = probabilities.rsample()#vuče random sample bez gradijenta - ne može da backpropagade
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)

        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - T.tanh(actions).pow(2) + self.reparameterization_noise)#uzima logaritamsku verovatnocu 
                                                                                      #jednog sample-a pod standardnom devijacijom

        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

def get_gradient_norm(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],env=None, gamma=0.99, n_actions=2,
                 max_size=1000000, tau = 0.005, layer1_size=256, layer2_size=256, batch_size=256, reward_scale=1.0):
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, max_action=env.action_space.high, n_actions=n_actions,name='actor')
        self.critic1 = CriticNetwork(beta,input_dims,layer1_size, layer2_size,n_actions=n_actions,name='critic1')
        self.critic2= CriticNetwork(beta,input_dims,layer1_size, layer2_size, n_actions=n_actions,name='critic2')
        self.value = ValueNetwork(beta,input_dims,layer1_size, layer2_size, name='value')
        self.target_value = ValueNetwork(beta, input_dims,layer1_size, layer2_size, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transitions(state,action,reward, new_state, done)

    def update_network_parameters(self, tau=None): #soft update value mreze
        if tau is None:
            tau = self.tau
        
        target_value_parameters = self.target_value.named_parameters()
        value_parameters = self.value.named_parameters()

        target_value_state_dict = dict(target_value_parameters)
        value_state_dict = dict(value_parameters)

        for name in value_state_dict:
            value_state_dict[name]=tau*value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(new_state).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward(state, actions)
        q2_new_policy = self.critic2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward()
        value_grad_norm = get_gradient_norm(self.value)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward(state, actions)
        q2_new_policy = self.critic2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = get_gradient_norm(self.actor)
        self.actor.optimizer.step()

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_#target qValue, ono cemu težimo
        q1_old_policy = self.critic1.forward(state, action).view(-1)
        q2_old_policy = self.critic2.forward(state, action).view(-1)
        critic1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()

        critic1_grad_norm = get_gradient_norm(self.critic1)
        critic2_grad_norm = get_gradient_norm(self.critic2)

        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_network_parameters()

        wandb.log({
            "critic_loss": critic_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "log_probs": log_probs.mean().item(),
            "actor_grad_norm": actor_grad_norm,
            "critic1_grad_norm": critic1_grad_norm,
            "critic2_grad_norm": critic2_grad_norm,
            "value_grad_norm": value_grad_norm,
            "actor_weights": wandb.Histogram(self.actor.fullyConnectedL1.weight.data.cpu()),
            "critic1_weights": wandb.Histogram(self.critic1.fullyConnectedL1.weight.data.cpu()),
            "value_weights": wandb.Histogram(self.value.fullyConnectedL1.weight.data.cpu())
        })



if __name__ == '__main__':
    training_period = 100
    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
    env = RecordVideo(env,
                video_folder="sac_training_video",
                name_prefix="training",
                episode_trigger=lambda x: x % training_period == 0  # Only record every 250th episode
                )
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    n_games = 1000
   
    wandb.init(
      project="bipedal-Ql",
      name=f"SAC",
      config={
      "algorithm": "SAC",
      "batch size":agent.batch_size,
      "alpha": agent.alpha,
      "gamma": agent.gamma,
      "layer 1 size":agent.layer1_size,
      "layer 2 size":agent.layer2_size,
      "episodes": n_games,
      })

    best_score = -1000
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, trunctured, info = env.step(action)
            done = terminated or trunctured
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        avg_score = np.mean(score_history[-100:])
        wandb.log({"episode": i, "score": score, "avg_score": avg_score})
        score_history.append(score)
        

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    video_folder = "./sac_training_video"
    video_files = glob.glob(os.path.join(video_folder, "*.mp4")) 

    for video_path in video_files:
        video_name = os.path.basename(video_path)
        wandb.log({
            video_name: wandb.Video(video_path,  format="mp4")
        })

    wandb.finish()
    env.close()