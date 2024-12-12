# built on cleanRL implementation of PPO
import os
import random
import time
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import pdb
from empyrical import max_drawdown,sharpe_ratio
from scipy.special import softmax
from PatchTST import Model as patchTST
import sys

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "portfolio_optimization"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    debug_run: bool = False
    """for testing purposes"""
    save_model: bool = False
    """whether to save model into the `{runs}/{run_name}` folder"""
    save_graph: bool = False
    """whether to save final graphs related to trained model"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "PortfolioEnv-v4"
    """the id of the environment"""
    dataset_file: str = "default"
    """name of the file which has stock market data, else generates [sin,cos]"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # agent related
    agent: str = 'nn'
    lag_enabled: bool = False
    num_layers: int = 1

    # env related
    balance: float = 100
    lag: int = 64 # 63 is number of trading days in a financial quarter
    k: float = 1
    render_step: bool = False
    dsr: bool = False
    dsr_scale: float = 0.004  # 0.5 for range of 0->1
    norm: bool = False
    mdd: bool = False
    shorting_allowed: bool = False
    softmax: bool = True

    # dataset generator related
    tts : float = 0.9
    n_days: int = 100
    n_assets: int = 5

    # running norms
    running_norm_obs: bool = False
    running_norm_reward: bool = False



def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        if args.running_norm_obs:
            env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -1000, 1000))
        if args.running_norm_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env = gym.wrappers.FrameStack(env,args.lag)
        if not args.lag_enabled:
            env = gym.wrappers.FlattenObservation(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PortfolioEnv4(gym.Env):
  def __init__(self,data=None,data_norm=None,balance=100,k=1,render_step=False,shorting_allowed=False):
    self.eps = 1e-12
    self.n = data.shape[0]
    self.n_assets = data.shape[1]
    self.n_stats = data.shape[2]
    self.data = np.float32(data)
    self.data_norm = data_norm #self._get_normalize()
    self.portfolio_value = 1
    self.init = balance
    self.balance = balance
    self.short_margin = 1
    self.long_margin = 1
    self.weights = np.array([0]*self.n_assets + [1])
    self.memory = {
        'weights' : [],
        'portfolio_value' : [],
        'balance' : [],
        'returns' : []
    }
    self.render_step = render_step
    self.memories = []
    self.k = k
    self.t = 0
    self.norm = args.norm
    self.sr = 0
    self.tc = 0.000 # transactional cost

    # dsr related
    self.A = 0
    self.B = 0
    # lag*asset*stats (lagged prices+stats) + asset(weight of assets) + 1(weight of cash) + 1(balance)
    self.observation_space = gym.spaces.Box(high = np.inf , low = -np.inf ,shape = (self.n_assets*self.n_stats+self.n_assets+2,1))
    self.action_space = gym.spaces.Box(low = -self.k if shorting_allowed else 0, high = self.k , shape = (self.n_assets,))

  def _get_obs(self):
    obs = np.array(list(self.data_norm[self.t:self.t+1,:,:].flatten())+list(self.weights.reshape(-1))+[self.balance/self.init])
    obs = np.clip(obs, np.finfo(np.float32).min, np.finfo(np.float32).max)
    return np.float32(obs)

  def reset(self,seed=None,options=None):
    super().reset(seed=seed)
    self.t = 0
    ep_ret = np.array(self.memory['returns']).reshape(-1)
    mdd = max_drawdown(ep_ret)
    sr = sharpe_ratio(ep_ret)
    tot_ret = (np.prod(1+ep_ret)-1)*100
    info = {
        'memory':self.memory,
        'mdd' : mdd,
        'sr' : sr,
        'ret' : tot_ret,
    }  # episodic sharpe,drawdown and returns

    self.memories.append(self.memory)
    self.memory = {
        'weights' : [],
        'portfolio_value' : [],
        'balance' : [],
        'returns': [],
    }
    self.portfolio_value = 1
    self.weights = np.array([0]*self.n_assets + [1])
    self.balance = self.init
    self.sr = 0
    return self._get_obs(),info

  '''
    diff -> return % of each asset
    w_ -> [(asset_index,action)]
    w -> weight of each asset
    profit return (r) ->  ( diff . w ) * balance
  '''
  def _get_value(self,w_,bal,short=False):
    if len(w_) == 0:
        return 0
    diff = np.array(list( (self.data[self.t+1,:,0] - self.data[self.t,:,0])/self.data[self.t,:,0] ))
    w = softmax([i for (_,i) in w_]) if args.softmax else weight_stack([i for (_,i) in w_])
    weights = np.zeros(diff.shape[0])
    for i,(j,_) in enumerate(w_):
       weights[j] = ( -1 if short else 1 ) * w[i]
    r = np.dot(weights,diff)*bal
    # print( f"r = {r} \n weights={weights} \n w={w} \n bal={bal} \n short = {short} \n w_ = {w_} ")
    if( np.isnan(r) or np.isinf(r) or np.isneginf(r) ):
        print('invalid r values enc ountered')
    return r

  def step(self,action):
    prev_obs = self._get_obs()
    wpos = 0
    wneg = 0
    w_p = []
    w_n = []
    for i in range(len(action)):
        if action[i]>0:
            wpos += action[i]
            w_p += [(i,action[i])]
        elif action[i]<0:
            wneg += action[i]
            w_n += [(i,-action[i])]

    if ( 1 - wpos + wneg ) > 0:
        w_bal = 1 - wpos + wneg
    else:
        w_bal = 0

    trade_bal = self.balance*(1-w_bal)
    long_bal = self.long_margin * trade_bal * (wpos)/(wpos-wneg+self.eps)
    short_bal = self.short_margin * trade_bal * (-wneg)/(wpos-wneg+self.eps)
    
    # print(f"action = {action}" )
    short_r = self._get_value(w_n,short_bal,True)
    long_r = self._get_value(w_p,long_bal)
    r = short_r + long_r - self.tc*trade_bal
    self.weights = np.array([w_bal] + list(action))

    returns = r/(self.balance+self.eps)
    self.balance = r + self.balance

    self.portfolio_value = self.balance/self.init
    self.memory['weights'].append(self.weights)
    self.memory['portfolio_value'].append(self.portfolio_value)
    self.memory['balance'].append(self.balance)
    self.memory['returns'].append(returns)

    # commented to save computation cost
    # sharpe_ratio = np.mean(self.memory['returns'])
    # sharpe_ratio = sharpe_ratio / (np.std(self.memory['returns'])+1e-9)


    reward = returns
    # max drawdown
    if args.mdd:
        mdd = max_drawdown(np.array(self.memory['returns']))
        reward = mdd

    # multiple NaN issues TODO
    if args.dsr:
        ret_hist = np.array(self.memory['returns'])
        if self.t < 20 :  # if more than 100 days use DSR else legacy St - St-1
            new_sr = sharpe_ratio(ret_hist)
            if np.isnan(new_sr) or self.t < 20 :
                dsr = returns * 20 * ( 2 * args.dsr_scale )# * (2*0.004) # estimated dsr at very small timestep
                # print('invalid new sr')
            else:
                dsr = new_sr - self.sr
            self.sr = new_sr
            self.A = np.mean(ret_hist[:-1])
            self.B = np.mean(ret_hist[:-1]**2)
        else:
            eta = 0.5 * ( 2* args.dsr_scale ) # * (2*0.004) # dsr scaler
            self.A = ( self.A * ( self.t - 1 ) + ret_hist[-2] ) / ( self.t ) # np.mean(ret_hist[:-1])
            self.B = ( self.B * ( self.t - 1 ) + ret_hist[-2]**2 ) / ( self.t ) # np.mean(ret_hist[:-1]**2)
            delta_A = returns - self.A
            delta_B = returns**2 - self.B
            Dt = (self.B*delta_A - 0.5*self.A*delta_B) / ((self.B-self.A**2)**(3/2)+self.eps)
            dsr = Dt*eta 
            if np.isnan(dsr):
                dsr = returns
                # print('invalid running dsr',ret_hist)
        #print(dsr)
        reward = dsr    # just for scaling

    if( self.render_step ):
      print(f" -------current-step--------")
      print(f"obs : {prev_obs.reshape(-1)}")
      print(f"action : {action.reshape(-1)}")
      print(f"weights : {self.weights.reshape(-1)}")
      print(f"reward : {r} , balance : {self.balance}, terminated : \
      { True if (self.t+1 == self.n or self.balance < -self.init*2 ) else False}, time = {self.t}")
      # print(f"MDD2 = {mdd} , sharpe : {sharpe_ratio}")
      print(f"---------------------------")


    self.t+=1
    terminated = True if (self.t+1 == self.n or self.balance < -self.init*2 ) else False

    # risk_sens = max(-20,-np.exp(-returns)) if returns < 0 else np.log(1+returns)
    info = {}
    return self._get_obs(), reward , terminated, False, info

class AgentNN(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

#DARNN encoder
class InputAttentionEncoder(nn.Module):
    def __init__(self, N, M, T, stateful=False):
        """
        :param: N: int
            number of time serieses
        :param: M:
            number of LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with values of the last cell state
            of previous time window or to initialize it with zeros
        """
        super(self.__class__, self).__init__()
        self.N = N
        self.M = M
        self.T = T
        
        self.encoder_lstm = nn.LSTMCell(input_size=self.N, hidden_size=self.M)
        
        #equation 8 matrices
        
        self.W_e = nn.Linear(2*self.M, self.T)
        self.U_e = nn.Linear(self.T, self.T, bias=False)
        self.v_e = nn.Linear(self.T, 1, bias=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    def forward(self, inputs):
        encoded_inputs = torch.zeros((inputs.size(0), self.T, self.M)).to(self.device)
        
        #initiale hidden states
        h_tm1 = torch.zeros((inputs.size(0), self.M)).to(self.device)
        s_tm1 = torch.zeros((inputs.size(0), self.M)).to(self.device)
        
        for t in range(self.T):
            #concatenate hidden states
            h_c_concat = torch.cat((h_tm1, s_tm1), dim=1)
            
            #attention weights for each k in N (equation 8)
            x = self.W_e(h_c_concat).unsqueeze_(1).repeat(1, self.N, 1)
            y = self.U_e(inputs.permute(0, 2, 1))
            z = torch.tanh(x + y)
            e_k_t = torch.squeeze(self.v_e(z))
        
            #normalize attention weights (equation 9)
            alpha_k_t = F.softmax(e_k_t, dim = 0 if len(e_k_t.shape)==1 else 1 )
            
            #weight inputs (equation 10)
            weighted_inputs = alpha_k_t * inputs[:, t, :] 
    
            #calculate next hidden states (equation 11)
            h_tm1, s_tm1 = self.encoder_lstm(weighted_inputs, (h_tm1, s_tm1))
            
            encoded_inputs[:, t, :] = h_tm1
        return encoded_inputs
    
# DA-RNN model
class AgentDARNN(nn.Module):
    def __init__(self, envs):
        super().__init__()
        W,E = envs.single_observation_space.shape
        self.n_stats = envs.get_attr('n_stats')[0]
        self.n_assets = envs.get_attr('n_assets')[0]
        self.encoder = InputAttentionEncoder(self.n_assets, 8, W, False)
        self.fc0 = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(W*E,32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 32)),
            nn.Tanh(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(W*8,32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 32)),
            nn.Tanh(),
            nn.Dropout(0.5),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
    
    def get_states(self, x):
        N,L,H_in = x.shape
        x1 = self.fc0(x)                # extract full info
        x = x[:,:,:self.n_assets]       # only stock time series
        x = self.encoder(x)
        x = self.fc(x)                  
        x = torch.cat((x,x1),dim=1)
        return x.reshape(N,-1)
    
    def get_value(self, x):
        hidden = self.get_states(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.get_states(x)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden) 

class AgentPatchTST(nn.Module):
    def __init__(self, envs):
        super().__init__()
        W,E = envs.single_observation_space.shape
        self.n_stats = envs.get_attr('n_stats')[0]
        self.n_assets = envs.get_attr('n_assets')[0]
        self.target_window = 32
        self.encoder = patchTST(self.target_window,W)
        self.fc0 = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(W*E, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 32)),
            nn.Tanh(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(self.target_window*self.n_assets,256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 32)),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
    
    def get_states(self, x):
        N,L,H_in = x.shape
        x1 = self.fc0(x)                # extract full info
        x = x[:,:,:self.n_assets]       # only stock time series
        x = self.encoder(x)
        x = self.fc(x)                  
        x = torch.cat((x,x1),dim=1)
        return x.reshape(N,-1)
    
    def get_value(self, x):
        hidden = self.get_states(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.get_states(x)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden) 

# evaluate function for saved model
def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    gamma: float = 0.99,
):  
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    sr = []
    ret = []
    mdd = []
    step_bal = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
            if 'sr' in infos:
                sr += infos['sr'].tolist()
                ret += infos['ret'].tolist()
                mdd += infos['mdd'].tolist() 
                step_bal += [infos['memory'][0]['balance']]
                print(f'final bal : {step_bal[-1][-1]} sharpe : {sr[-1]} return : {ret[-1]} max drawdown {mdd[-1]}')
        obs = next_obs

    return episodic_returns,sr,ret,mdd,step_bal


# magnitude based normalization
def get_normalize( data ):
  data_norm = np.array(data)
  for i in range(data.shape[1]):
    for j in range(data.shape[2]):
      div = 10**((np.log10(1+np.mean(abs(data[:,i,j])))+1)//1) # will make range [-10,10]
      data_norm[:,i,j] = (data[:,i,j]) / (div)
  return data_norm

def weight_stack(w):
    w_sum = np.sum(w) # catch error
    return w/w_sum

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    runs = 'debug_runs' if args.debug_run else 'runs'
    writer = SummaryWriter(f"{runs}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    if args.dataset_file == 'default':
        import sys
        sys.exit("you have to enter a file name")
    else:
        dataset = np.load(args.dataset_file)

    plt.plot(dataset[:,:,0])
    plt.title('Dataset')
    plt.savefig(f'{runs}/{run_name}/dataset.png')
    plt.clf()

    if args.norm:
        dataset_norm = get_normalize(dataset)
    else:
        dataset_norm = np.array(dataset)
    tts = args.tts
    train_dataset = dataset[:int(tts*len(dataset))]
    train_dataset_norm = dataset_norm[:int(tts*len(dataset))]
    test_dataset = dataset[int((tts)*len(dataset)):]
    test_dataset_norm = dataset_norm[int((tts)*len(dataset)):]

    if args.agent == 'nn':
        Agent = AgentNN
    elif args.agent == 'darnn':
        Agent = AgentDARNN
        args.lag_enabled = True
    elif args.agent == 'patchtst':
        Agent = AgentPatchTST
        args.lag_enabled = True

    gym.envs.register(
        id='PortfolioEnv-v4',
        entry_point='__main__:PortfolioEnv4',
        max_episode_steps= len(train_dataset)-1,
        kwargs={'data' : train_dataset, 'data_norm': train_dataset_norm, 'balance' : args.balance, 'k' : args.k, 'shorting_allowed': args.shorting_allowed },
    )
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    avg_bal = 0

    gym.envs.register(
            id='PortfolioEnv-Testv4',
            entry_point='__main__:PortfolioEnv4',
            max_episode_steps= len(test_dataset)-1,
            kwargs={'data' : test_dataset, 'data_norm': test_dataset_norm, 'balance' : args.balance, 'k' : args.k, 'shorting_allowed': args.shorting_allowed }
    )
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                if 'sr' in infos:
                    writer.add_scalar("charts/avg_episodic_sharpe", np.mean(infos["sr"]), global_step)
                    writer.add_scalar("charts/avg_episodic_total_return", np.mean(infos["ret"]), global_step)
                    writer.add_scalar("charts/avg_episodic_MaxDrawDown", np.mean(infos["mdd"]), global_step)

                num_test_episodes = 10
                model_path = f"{runs}/{run_name}/{args.exp_name}.cleanrl_model0"
                torch.save(agent.state_dict(), model_path)
                print(f"model/2 saved to {model_path}")
                episodic_returns,sr,ret,mdd,step_bal = evaluate(
                    model_path,
                    make_env,
                    'PortfolioEnv-Testv4',
                    eval_episodes=num_test_episodes,
                    run_name=f"{run_name}-eval",
                    Model=Agent,
                    device=device,
                    gamma=args.gamma,
                )
                balances = np.array(step_bal)
                cur_avg_bal = np.mean(balances[:,-1])
                if avg_bal <= cur_avg_bal:
                    print(f"best model {model_path}.best")
                    torch.save(agent.state_dict(), model_path+'.best')
                    avg_bal = cur_avg_bal

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    
    if args.save_graph:
        model_path = f'{runs}/{run_name}/'
        
        # different env final performance plot
        for i in range(args.num_envs):
            plt.plot(envs.get_attr('memories')[i][-1]['balance'],label=f'{i}')
        plt.legend()
        plt.savefig(model_path+'envs.png')

        # best result env performance in 1st , 1/4th , 1/2nd, 3/4th and final episodes
        memories = envs.get_attr('memories')[0]
        plt.figure(figsize=(20,5))
        tot_eps = len(memories)
        if tot_eps > 1:
            plt.plot(memories[1]['balance'],label='episode 1')
            plt.plot(memories[tot_eps//4]['balance'],label=f'episode {tot_eps//4}')
            plt.plot(memories[tot_eps//2]['balance'],label = f'episode {tot_eps//2}')
            plt.plot(memories[3*tot_eps//4]['balance'],label = f'episode {3*tot_eps//4}')
            plt.plot(memories[-1]['balance'], label = f'last episode {tot_eps}')
            plt.legend()
            plt.savefig(model_path+'env_reward.png')

            # plot the weight distributions of assets for final episode
            n_assets = envs.get_attr('n_assets')[0]+1
            idx = tot_eps-1
            df = pd.DataFrame(
                    {f'w{i}': np.array(memories[idx]['weights'])[:,i].reshape(-1) for i in range(n_assets)},
                    index = np.arange(0,len(np.array(memories[idx]['weights'])[:,0].reshape(-1)),1)
                )
    if args.save_model:
        model_path = f"{runs}/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        
        num_test_episodes = 10
        episodic_returns,sr,ret,mdd,step_bal = evaluate(
            model_path,
            make_env,
            'PortfolioEnv-Testv4',
            eval_episodes=num_test_episodes,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
            writer.add_scalar("eval/sharpe_ratio", sr[idx], idx)
            writer.add_scalar("eval/rate_of_return", ret[idx], idx)
            writer.add_scalar("eval/mdd", mdd[idx], idx)

        balances = np.array(step_bal)
        min_idx = np.argmin(balances[:,-1])
        max_idx = np.argmax(balances[:,-1])
        avg_idx = np.argsort(balances[:,-1])[len(balances[:,-1])//2]

        for i in range(len(step_bal[0])):
            writer.add_scalar('eval/min_stepwise_balance',step_bal[min_idx][i],i)
            writer.add_scalar('eval/max_stepwise_balance',step_bal[max_idx][i],i)
            writer.add_scalar('eval/avg_stepwise_balance',step_bal[avg_idx][i],i)

    envs.close()
    writer.close()
'''
---- basic run ---- 

python3 cleanrl/pf4.py \
    --norm \
    --dsr \
    --dataset-file "dataset_path" \
    --agent "patchtst" \
    --save-graph \
    --num-envs 16 \
    --lag 64 \
    --num-steps 256 \
    --save-model \
    --total-timesteps 100000
'''
