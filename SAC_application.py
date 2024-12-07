import torch
from envs.build_env import get_instance_env
from models.SAC_model import Actor_Net,Double_Q_Net
from SAC_collect import collector
from SAC_learn import Learner
from logs.mylog import logger,Wandb_log
from  torch import multiprocessing as _mp
from torch.nn import functional as F
from SAC_evaluate import model_evalutor
import warnings
warnings.filterwarnings('ignore')
import time
from buffer import ReplayBuffer






torch.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

class SAC:

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
        self.action_shape = self.env.action_space.n
        self.obs_shape = self.env.observation_space.shape
        self.state_dim = self.obs_shape[0]


        #生成神经网络模型
        self.actor_target = Actor_Net(self.state_dim, self.action_shape)


        logger.info('神经网络启动成功')



        #加载断点数据
        if model_state_dict is not None:
            self.actor_target.load_state_dict(torch.load(model_state_dict))
        self.actor_target.share_memory()
        print(self.actor_target)

        


    
    def train(self):

        

        #数据库
        device = 'cuda:0'
        self.database = ReplayBuffer(state_dim =self.obs_shape)
        self.database.share_memory()

        #数据采集
        mp = _mp.get_context("spawn")
        collect_env_num = 10
        process1 = mp.Process(target=collector, args=(
                                                    self.env, 
                                                    self.actor_target,
                                                    collect_env_num,
                                                    self.database,
                                                    device))
        process1.start()
        logger.info('初始化数据采集')

     

        #模型训练
        lr=1e-2
        batch_size = 512
        gamma = 0.9
        alpha = 0.2
        adaptive_alpha = True
        device = 'cuda'

        learner = Learner(self.actor_target,self.state_dim,self.action_shape,lr, batch_size,gamma,alpha,adaptive_alpha,device)
        process2 = mp.Process(target=learner, args=(self.database,))
        process2.start()


        logger.info('初始化模型训练')


        #模型评估
        process3 = mp.Process(target=model_evalutor, args=(self.actor_target,self.env_id,self.world,self.stage))
        process3.start()
        logger.info('初始化评估器')




        count  = 0
        while True:
            # count += 1
            print(self.database.size)
            time.sleep(1)
            # print('主程序中：')
            # print(self.database.s.is_shared())
            # print(self.database.s.data.data_ptr())
            # log_info['train_iter'] =curr_episode
            

            # if curr_episode % save_model_freq == 0:
            #     ckpt_path = r'D:\reinforce\mario\my_mario\ckpt\{}_iteration.pth.tar'.format(curr_episode)
            #     torch.save(self.model.state_dict(),ckpt_path)


            

if __name__ == '__main__':
    agent = SAC(env_id = "mario")
    agent.train()
    