#import
import os
import numpy as np
from scipy import stats

size_set  = [25, 50, 100, 200, 400, 800]
datatype_set = ['DR5','DR10','OR']
dataname_set = [['abalone-5','bank1-5','bank2-5','calhousing-5','census1-5','census2-5','computer1-5','computer2-5'], 
                ['abalone-10','bank1-10','bank2-10','calhousing-10','census1-10','census2-10','computer1-10','computer2-10'], 
                ['car','ERA','LEV','SWD','winequality-red']]


method = 'SL'
task = 'P'
size = 800

for a in [0,1,2]:
    for b in [0,1,2,3,4,5,6,7]:
        if a!=2 or b<=4:
            name = dataname_set[a][b]
            ME = np.zeros(100)
            if os.path.isfile("./Results/%s/%s/%s-%d.csv"%(method,task,name,size)):
                data = np.loadtxt("./Results/%s/%s/%s-%d.csv"%(method,task,name,size),delimiter = ",")
                ME = data[:100,3]
            print(name, "\n$%.4f_{%.4f}$"%(np.mean(ME),np.std(ME)))


for a in [0,1,2]:
    for b in [0,1,2,3,4,5,6,7]:
        if a!=2 or b<=4:
            name = dataname_set[a][b]
            ME = np.zeros(100)
            if os.path.isfile("./Results/%s/%s/%s-%d.csv"%(method,task,name,size)):
                data = np.loadtxt("./Results/%s/%s/%s-%d.csv"%(method,task,name,size),delimiter = ",")
                ME = data[:100,4]
            print(name, "\n$%.4f_{%.4f}$"%(np.mean(ME),np.std(ME)))
