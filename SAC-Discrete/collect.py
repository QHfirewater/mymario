import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions import Categorical
from envs.env_manage import MultiEnvManager
from models.SAC_model import Actor_Net
from envs.build_env import get_instance_env


def collector(actor_target,database):

    env = get_instance_env(env_id='LunarLander-v2',world=1,stage=1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor_Net(state_dim, action_dim)
   


    while True:

        s = env.reset()  # Do not use opt.seed directly, or it can overfit to opt.seed
        done = False

        while not done:
			#e-greedy exploration
            if database.size < 1e4:
                a = env.action_space.sample()
            else:
                state = torch.from_numpy(s) #from (s_dim,) to (1, s_dim)
                probs = actor(state)
                a = Categorical(probs).sample().item()

            s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated
            done = (dw or tr)
            database.add(s, a, r, s_next, dw)
            s = s_next

            actor.load_state_dict(actor_target.state_dict())
   
