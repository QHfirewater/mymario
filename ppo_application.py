
import torch
from envs.build_env import get_instance_env
from models.model import Model
from collect import collector
from learn import Multistep_learn
from evaluate import model_evalutor

from logs.mylog import logger,Wandb_log
import time
from  torch import multiprocessing as _mp
from torch.nn import functional as F
import warnings
warnings.filterwarnings('ignore')



torch.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

class PPO:

    def __init__(self,env_id= None,world  =1,stage = 1,model_state_dict= None) -> None:


        
        #处理程序的配置问题
        assert env_id is not None , "Please specify env_id"
        self.env_id = env_id

        

        #生成训练的环境
        self.world = world
        self.stage = stage
        self.env = get_instance_env(env_id,self.world,self.stage,)
        logger.info('环境启动成功!')

        
        #读取环境中动作空间形状
        action_shape = self.env.action_space.n
        obs_shape = self.env.observation_space.shape


        #生成神经网络模型
        self.model = Model(obs_shape[0], action_shape)
        self.model.share_memory()
        
    
        logger.info('神经网络启动成功')


        if model_state_dict is not None:
            self.model.load_state_dict(torch.load(model_state_dict))
        print(self.model)

        

    def train(self):


        
        mp = _mp.get_context("spawn")
        collect_pare_pipe,collect_child_pipe = mp.Pipe()


        collect_steps = 512
        gamma = 0.9
        tau = 1
        device = 'cuda:0'
        collect_env_num = 10
        process = mp.Process(target=collector, args=(collect_child_pipe,
                                                    self.env, 
                                                    self.model,
                                                    collect_env_num,
                                                    collect_steps,
                                                    gamma,
                                                    tau,
                                                    device))
        process.start()
        logger.info('初始化数据采集')



        lr=5e-5
        num_per_epochs = 10 
        device = 'cuda:0'
        epsilon = 0.2
        beta = 0.01
        batch_size = 512
        learner = Multistep_learn(self.model,lr,num_per_epochs,device,epsilon,beta,batch_size)
        logger.info('初始化模型训练')


        #生成评估器
        mp = _mp.get_context("spawn")
        # eval_pare_pipe,eval_child_pipe = mp.Pipe()
        process = mp.Process(target=model_evalutor, args=(self.model,self.env_id,self.world,self.stage))
        process.start()

        logger.info('初始化评估器')
    
    
        #生成监视器
        record = Wandb_log(project_name='mario44',model=self.model)
    
        
        curr_episode = 0
        save_model_freq = 20
        while True:
            #生成数据
            timesteps = collect_pare_pipe.recv()


            #模型迭代

            log_info = learner(timesteps)
            # collect_pare_pipe.send(self.model.state_dict())
            # self.model.load_state_dict(model_state)




            curr_episode +=1
            log_info['train_iter'] =curr_episode
            


            if curr_episode % save_model_freq == 0:
                ckpt_path = r'D:\reinforce\mario\my_mario\ckpt\{}_iteration.pth.tar'.format(curr_episode)
                torch.save(self.model.state_dict(),ckpt_path)


            #记录过程
            record(log_info)
            print(log_info)
            
            
        

    def deploy(self,model_state_dict,video_path):
        
        if model_state_dict is not None:
            self.model.load_state_dict(torch.load(model_state_dict))
        logger.info('模型加载成功')

        self.video_path = video_path

        env = get_instance_env(self.env_id, self.world,self.stage,self.video_path)

        obs = env.reset()
    
    
        logger.info('环境创建成功')
        while True:
            self.model.eval()
            obs = torch.from_numpy(obs)
            logits,value = self.model(obs)
            policy = F.softmax(logits)
            action  = policy.argmax(dim=-1)
            obs,reward,done,info = env.step(action.item())
            env.render()
            time.sleep(0.1)
            print('x:',info['x_pos'],'y:',info['y_pos'])
            if done:
                print(info['x_pos'],info['y_pos'])
                break




        

        


