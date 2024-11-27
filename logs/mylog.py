import logging
import wandb
import torch

#程序log
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fh = logging.FileHandler('test.log')
fh.setFormatter(fmt=formatter)

ch= logging.StreamHandler()
ch.setFormatter(fmt=formatter)


logger.addHandler(fh)
logger.addHandler(ch)



#数据log
class Wandb_log:
    def __init__(self,project_name,model) -> None:
        wandb.init(project=project_name, reinit=True)

        wandb.watch(model, log="all", log_freq=100, log_graph=True)
        
    def __call__(self, info_for_log):

        wandb.log(data=info_for_log, step=info_for_log['train_iter'])



if __name__ == "__main__":
    wandb.init(project='test',reinit=True)
    
    a = torch.randint(3,5,size=(1,))
    info = {"a":a,
            "b":1}
    wandb.log(data=info,step=info['b'])
    