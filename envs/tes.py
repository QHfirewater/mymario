import torch
from torch import multiprocessing as mp
import time 



# def fun1(t):
#     print("process begin 2")
#     t_ = t.cuda()
#     print('子进程中t:',t)
#     print('子进程中t:',t.is_shared())
    
#     t_ += 10
#     # t += t_.cpu()
#     t = 300
#     print('子进程中t:',t)
#     print("process end")


def fun2(t):

    print("process begin 1")
    print('子进程中t:',t)
    print('子进程中t:',t.is_shared())
    
    

    t = t + 0.1
    
    print('子进程中t:',t)
    print(id(t))
    print("process end")
    time.sleep(1)


if __name__ == "__main__":
    a = torch.tensor([.3, .4, 1.2])
    a = a.cuda('cuda:0')
    
    a.share_memory_()
    print('主程序中a:',a)
    print('主程序中11：',a.is_shared())
    


    # a = torch.tensor([20.3, .4, 1.2])
    # process = []
    # p = mp.Process(target=fun1, args=(a, ))
    # p.start()
    # p.join()

    p = mp.Process(target=fun2, args=(a, ))
    p.start()
    p.join()


    a += 10
    print(id(a))
    print('主进程中t:',a)
    print('主程序中11：',a.is_shared())


