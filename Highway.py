import gymnasium as gym
import ray
import highway_env
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.distributions import Categorical, MultivariateNormal
# from gym.wrappers.monitoring.video_recorder import VideoRecorder
#ray.init()
import os


# from pyvirtualdisplay import Display

# display = Display(visible=False, size=(1400, 900))
# _ = display.start()

path = os.path.abspath(__file__)
before_training = os.path.join(path.rsplit('/',1)[0], "before_training.mp4")



gamma = 0.95
alpha_actor = 0.01
alpha_critic = 0.001
episode_len = 100
update_interval = 5
#Setting up the Gym environment highway-fast
env = gym.make("highway-fast-v0", render_mode='human')
config = {
       "observation": {
           "type": "GrayscaleObservation",
           "observation_shape": (128, 64),
           "stack_size": 4,
           "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
           "scaling": 1.75,
       },
       "policy_frequency": 2
   }

env.configure(config)
# video = VideoRecorder(env, before_training, enabled=True)

obs, info = env.reset()
done = truncate = False
actor_output_dim = env.action_space.n
critic_output_dim = 1
input_dims = env.observation_space.shape[0]

class actor(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(actor, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 2)
        self.linear1 = nn.Linear(32*31*15, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, output_dim)    
    
    def forward(self, x):
     #   print(x.size())
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
       # x = x-torch.max(x)
        prob = F.softmax(x, dim=-1)
        return prob
    
    def num_flat_features(self, x):
     #   print(x.size())
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class critic(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(critic, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 2)
        self.linear1 = nn.Linear(32*31*15, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)    
    
    def forward(self, x):
     #   print(x.size())
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
       # x = x-torch.max(x)
        #x = F.softmax(x, dim=-1)
        return x
    
    def num_flat_features(self, x):
     #   print(x.size())
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class PPO:
    global   actor_output_dim, critic_output_dim, gamma, alpha_actor, alpha_critic, update_interval,env
    def __init__(self, input_dim, alpha_actor=0.01, alpha_critic=0.001, batch_size=4 ):
 
        self.actor = actor(input_dim, actor_output_dim)
        self.critic = critic(4, critic_output_dim)
        try:
            pretrained_weights_actor = torch.load('./ppo_actor.pth')
            pretrained_weights_critic = torch.load('./ppo_critic.pth')
            self.actor.load_state_dict(pretrained_weights_actor)
            self.critic.load_state_dict(pretrained_weights_critic)
            print('weights loaded')
        except:
            pass
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=alpha_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=alpha_critic)
        self.time_steps = 0
        self.clip = 0.5
        self.batch_size = batch_size
   
    
    def learn(self, max_time_steps):
        self.time_steps = 0
        time_steps_current = 0
        iteration_num = 0
        while self.time_steps < max_time_steps:
            batch_obs, batch_action, batch_reward, batch_eps_len, batch_log_prob, batch_eps_done = self.rollout()
            #print(batch_reward.shape)
            returns, advantage = self.compute_returns_and_advantages(batch_reward, batch_obs, batch_eps_done)
            iteration_num += 1
            time_steps_current = torch.sum(batch_eps_len)
            print(batch_obs.shape)
            V, _ = self.evaluate(batch_obs=batch_obs, batch_actions=batch_action)
            # print(V.shape)
           # advantage = returns - V.detach()
            
            #advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
            for _ in range(1):
                V, curr_prob_dist = self.evaluate(batch_obs=batch_obs, batch_actions=batch_action)
                
                #what is current prob dist and batch log prob?
                #current prob dist: #get action prob from changed network and get the action based on probability
                #batch log prob: get the action based on old network and get the action based on probability
             
                ratios = torch.exp(curr_prob_dist - batch_log_prob.squeeze()) #Expectation( pi-theta(a_t|s_t)/pi_theta_old(a_t|s_t))
                #returns, advantage = self.compute_returns_and_advantages(batch_reward, batch_obs)
                print(ratios.shape)
                l1 = ratios*advantage
                l2 = torch.clamp(ratios, 1-self.clip, 1+self.clip)*advantage
                
                actor_loss = -torch.min(l1,l2).mean()
            
                print(actor_loss)
                critic_loss = nn.MSELoss()(returns, V)
                print(critic_loss)
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                # for name, parameters in self.actor.named_parameters():
                #     print(name, parameters)
                
                # print(actor_loss)
                # print(critic_loss)
         # Save our model if it's time
            print(iteration_num)
            if iteration_num % 10 == 9:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')
            
     
    def rollout(self):
        batch_obs = []
        batch_action = []
        batch_reward = []
        batch_log_prob = []
        batch_eps_len = []
        batch_eps_done = []
        
        eps_reward =[]
        batch_num = 0
        
        while batch_num < self.batch_size:
            
            obs, info = env.reset()
            done = truncate = False
            reward_episode = 0
            eps_len = 0
            batch_num += 1
            eps_reward =[]
            
            while not (done): # or truncate
                batch_obs.append(obs)
                obs = torch.from_numpy(obs).to(torch.float).unsqueeze(0)
                # action = torch.argmax(self.actor(obs))
                action_val = self.actor(obs)
                #print(action_val.shape) 
                action_dist = Categorical(action_val)
                action = action_dist.sample()
              #  print('-'*50)
                #print(action)
              #  print(action_val)
                action_logProb = action_dist.log_prob(action)
               # print(action_logProb)
                batch_action.append(action.detach().numpy())
                # print(torch.sum(val))  
                batch_log_prob.append(action_logProb)
               # print(action_val)
                # env.render()
                # video.capture_frame()
                # print(action)
                obs, reward, done, truncate, info = env.step(action.detach().item())
                # batch_obs.append(obs)
                eps_reward.append(reward)
                batch_eps_done.append(done)
                # reward_episode = reward + gamma*reward_episode*(1-done)
                eps_len += 1
                
                # if batch_num % 10 == 9:
                    
                #     env.render()
                if done or truncate:
                    
                    batch_reward.append(eps_reward)
                   # print(np.array(eps_reward).shape, np.array(batch_reward).shape)
                    batch_eps_len.append(eps_len)
                    break
        print(f"reward for episodes: {sum(eps_reward)/batch_num}")
        batch_action = np.stack(batch_action)
        batch_action = torch.tensor(batch_action, dtype=torch.float)
       # batch_reward = torch.tensor(batch_reward, dtype=torch.float)
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_eps_len = torch.tensor(batch_eps_len, dtype=torch.float)
        batch_log_prob = torch.tensor(batch_log_prob, dtype=torch.float)
        return batch_obs, batch_action, batch_reward, batch_eps_len, batch_log_prob, batch_eps_done
            
    def evaluate(self, batch_obs, batch_actions):
        #print(batch_actions.size(), batch_obs.size())
        
        V = self.critic(batch_obs).squeeze()
        actor_action_prob = self.actor(batch_obs)
        # print(actor_action_prob.shape)
        #dist = MultivariateNormal(actor_action_prob, self.cov_mat)
        #action_log_prob = dist.log_prob(batch_actions)

        dist = Categorical(actor_action_prob)
        # # print(f'batch_actions {batch_actions.squeeze().shape}')
        action_log_prob = dist.log_prob(batch_actions.squeeze())
        #action_log_prob = torch.argmax(actor_action_prob)
        
        return V, action_log_prob# action_log_prob
    
    def compute_returns_and_advantages(self, batch_rewards, batch_obs, batch_done, gamma=0.99, lamda=0.95):
        returns = []
        advantages = []
        R = 0  # Value function estimate for the last state
       # print(batch_rewards.shape)
        done_counter = 0
        for rewards in reversed(batch_rewards):
            R = 0  #if batch_done[done_counter] else R# Value function estimate for the last state
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            done_counter += 1
        
        values = self.critic(batch_obs).squeeze()  # Get value function estimates for states
        #[[1,2,3],[4,5,6]]
        value_counter = 0
        
        for i in range(len(batch_rewards)):
            advantage = 0
            for t in range(len(batch_rewards[i])):
                if value_counter + 1 < len(values):
                    delta = batch_rewards[i][t] + gamma* values[value_counter+1]* (1 - batch_done[value_counter])- values[value_counter] #gamma * values[value_counter+1]*(1 - batch_done[value_counter])
                else:
                    delta = batch_rewards[i][t] + gamma *values[value_counter] *(1 - batch_done[value_counter] - values[value_counter])#dd- values[value_counter]
                advantage = delta + gamma * lamda * advantage * (1 - batch_done[value_counter])
                advantages.append(advantage)
                value_counter += 1
      
        returns = torch.tensor(returns, dtype=torch.float)
        advantages = torch.tensor(advantages, dtype=torch.float)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages
        
        return returns, gamma*advantages+values

                    
# video.close()
# video.enable = False


ppo = PPO(input_dim=4, alpha_actor=alpha_actor, alpha_critic=alpha_critic)

ppo.learn(max_time_steps=100000000)
env.close()

    