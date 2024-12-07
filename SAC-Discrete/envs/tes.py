import torch
from torch import nn
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



def collector(t):

    print("process begin 1")
    print('子进程中t:',t)
    print('子进程中t:',t.is_shared())
    
    

    t = t + 0.1
    
    print('子进程中t:',t)
    print(t.data_ptr())
    print("process end")
    time.sleep(1)


if __name__ == "__main__":
    # a = torch.tensor([.3, .4, 1.2])
    # a = a.cuda('cuda:0')
    
    # a.share_memory_()
    # print('主程序中a:',a.data_ptr())
    # print('主程序中11：',a.is_shared())


    test = torch.rand(size=(20,)).to('cuda:0')
    test.share_memory_()
    print('test',test.data_ptr())
    print(test.device)


    process = mp.Process(target=collector, args=(
                                    
                                                    test,
                                                    ))
    
    process.start()
    process.join()
    


    # a = torch.tensor([20.3, .4, 1.2])
    # process = []
    # p = mp.Process(target=fun1, args=(a, ))
    # p.start()
    # p.join()

    # p = mp.Process(target=fun2, args=(a, ))
    # p.start()
    # p.join()


    # a += 10

    # print('主进程中t:',a)
    # print('主程序中11：',a.is_shared())
    # print('主程序中a:',a.data_ptr())


