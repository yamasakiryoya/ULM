#import
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from numpy import mean,std
from sklearn.model_selection import train_test_split
from multiprocessing import Pool,Process,Pipe,Manager
from setproctitle import setproctitle
import MyFunc as MF

#parameter
size_set  = [25, 50, 100, 200, 400, 800]
batc_set  = [4, 8, 16, 32, 64, 128]
node_set  = [100]
TR, MP, EP, IS = 100, 10, 1000, 1
datatype_set = ['DR5','DR10','OR','SY']
dataname_set = [['abalone-5','bank1-5','bank2-5','calhousing-5','census1-5','census2-5','computer1-5','computer2-5'], 
                ['abalone-10','bank1-10','bank2-10','calhousing-10','census1-10','census2-10','computer1-10','computer2-10'], 
                ['car','ERA','LEV','SWD','winequality-red']]

#dataset
args = sys.argv
method, A, B, C = args[1], int(args[2]), int(args[3]), int(args[4])
data_path = "../datasets/"+datatype_set[A]+"/"+dataname_set[A][B]+".csv"
trte_data = np.loadtxt(data_path, delimiter = ",")
samplenum = trte_data.shape[0]
dimension = trte_data.shape[1]-1
classnum  = int(np.max(trte_data[:,-1])-np.min(trte_data[:,-1])+1)
print(dataname_set[A][B], size_set[C], samplenum, dimension, classnum)

#learning function
def learning(seed, train_data, test_data, node):
    #set seed, device
    torch.manual_seed(seed)
    device = torch.device('cpu')
    #arrange dataset
    train_X, test_X = torch.FloatTensor(train_data[:,:-1]), torch.FloatTensor(test_data[:,:-1])
    train_Y, test_Y = torch.LongTensor(train_data[:,-1]),   torch.LongTensor(test_data[:,-1])
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=batc_set[C], shuffle=True, drop_last=True)
    test_loader  = DataLoader(TensorDataset(test_X,  test_Y),  batch_size=len(test_Y))
    #set model, optimizer
    model = eval("MF."+method.replace('-','_'))(dimension, node, classnum).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.1**4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=(10.**4)**(1./EP))
    #learning
    results = np.zeros((EP//IS,11))
    for e in range(1,EP+1):
        MF.train_func(model, device, train_loader, optimizer)
        if e%IS == 0:
            results[e//IS-1,0]  = e
            results[e//IS-1,1:] = MF.test_func(model, device, test_loader, classnum, 'CV')
        scheduler.step()
    return results

#traintest function
def traintest(seed, results):
    #load dataset
    train_data, test_data = train_test_split(trte_data, train_size=size_set[C], random_state=seed, stratify=trte_data[:,-1])
    #traintest
    tmp = np.zeros((len(node_set),EP//IS,11))
    tmpP, tmpZ, tmpA, tmpS = np.zeros((len(node_set),11)), np.zeros((len(node_set),11)), np.zeros((len(node_set),11)), np.zeros((len(node_set),11))
    for i in range(len(node_set)):
        setproctitle("%s:%d:%d:%d:%d:%d"%(method, A, B, C, seed, i))
        print(dataname_set[A][B], size_set[C], seed, i)
        tmp[i,:,:] = learning(seed, train_data, test_data, node_set[i])
        tmpP[i,:] = tmp[i,MF.back_argmin(tmp[i,:, 1]),:]
        tmpZ[i,:] = tmp[i,MF.back_argmin(tmp[i,:, 2]),:]
        tmpA[i,:] = tmp[i,MF.back_argmin(tmp[i,:, 6]),:]
        tmpS[i,:] = tmp[i,MF.back_argmin(tmp[i,:,10]),:]
    #summary
    res  = np.zeros(48)
    parP = MF.back_argmin(tmpP[:, 1]); res[ 0] = parP; res[ 1:12] = tmpP[parP,:]
    parZ = MF.back_argmin(tmpZ[:, 2]); res[12] = parZ; res[13:24] = tmpZ[parZ,:]
    parA = MF.back_argmin(tmpA[:, 6]); res[24] = parA; res[25:36] = tmpA[parA,:]
    parS = MF.back_argmin(tmpS[:,10]); res[36] = parS; res[37:48] = tmpS[parS,:]
    results[seed] = res.tolist()
    print(res)

#parallel processing
def parallel():
    manager = Manager()
    jobs, results = [], manager.list(range(TR))
    for i in range(int(TR//MP)):
        for seed in range(i*MP,(i+1)*MP):
            job = Process(target=traintest, args=(seed, results))
            jobs.append(job)
            job.start()
        for job in jobs:
            job.join()
    tmp = np.array(results)
    print(tmp)
    res = np.zeros((int(TR+2), tmp.shape[1]))
    res[:TR,:], res[TR,:], res[int(TR+1),:] = tmp, mean(tmp,axis=0), std(tmp,axis=0)
    if os.path.isdir("./Results/%s/P"%method)==False: os.makedirs("./Results/%s/P"%method)
    if os.path.isdir("./Results/%s/Z"%method)==False: os.makedirs("./Results/%s/Z"%method)
    if os.path.isdir("./Results/%s/A"%method)==False: os.makedirs("./Results/%s/A"%method)
    if os.path.isdir("./Results/%s/S"%method)==False: os.makedirs("./Results/%s/S"%method)
    np.savetxt("./Results/%s/P/%s-%d.csv"%(method,dataname_set[A][B],size_set[C]), res[:, 0:12], delimiter=",")
    np.savetxt("./Results/%s/Z/%s-%d.csv"%(method,dataname_set[A][B],size_set[C]), res[:,12:24], delimiter=",")
    np.savetxt("./Results/%s/A/%s-%d.csv"%(method,dataname_set[A][B],size_set[C]), res[:,24:36], delimiter=",")
    np.savetxt("./Results/%s/S/%s-%d.csv"%(method,dataname_set[A][B],size_set[C]), res[:,36:48], delimiter=",")

#main function
if __name__ == "__main__":
    parallel()
