import logging
import wandb

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



