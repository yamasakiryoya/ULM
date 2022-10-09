#import
import os
import numpy as np
import pandas as pd
from scipy import stats

size_set  = [25, 50, 100, 200, 400, 800]
datatype_set = ['DR5','DR10','OR']
dataname_set = [['abalone-5','bank1-5','bank2-5','calhousing-5','census1-5','census2-5','computer1-5','computer2-5'], 
                ['abalone-10','bank1-10','bank2-10','calhousing-10','census1-10','census2-10','computer1-10','computer2-10'], 
                ['car','ERA','LEV','SWD','winequality-red']]


#table 3
method_set = ['BIN','POI','PO-ORD-ACL','PO-VS-SL',
              'OH-BIN','OH-POI','OH-ORD-ACL','OH-VS-SL',
              'ORD-ACL','VS-SL','ACL','SL']

for task in ['P']:
    if task=='P':
        print("NLL")
    else:
        print("\nM%sE"%task)
    res = np.zeros((6,len(method_set),3))
    for c in [0,1,2,3,4,5]:
        size = size_set[c]
        for a in [0,1,2]:
            for b in [0,1,2,3,4,5,6,7]:
                if a!=2 or b<=4:
                    name = dataname_set[a][b]
                    ME = np.zeros((len(method_set),100))
                    for method in method_set:
                        if os.path.isfile("./Results/%s/%s/%s-%d.csv"%(method,task,name,size)):
                            data = np.loadtxt("./Results/%s/%s/%s-%d.csv"%(method,task,name,size),delimiter = ",")
                            if task=='P': ME[method_set.index(method),:] = data[:100,2]
                            if task=='Z': ME[method_set.index(method),:] = data[:100,3]
                            if task=='A': ME[method_set.index(method),:] = data[:100,7]
                            if task=='S': ME[method_set.index(method),:] = data[:100,11]
                    for i in range(len(method_set)):
                        for j in range(len(method_set)):
                            if i!=j:
                                test = stats.mannwhitneyu(ME[i,:],ME[j,:], alternative='less')
                                #i win, j lose
                                if   test.pvalue<=0.05:
                                    res[c,i,0]+=1
                                    res[c,j,2]+=1
                                else:
                                    res[c,i,1]+=1
                                    res[c,j,1]+=1
    for i in range(len(method_set)):
        for c in [0,1,2,3,4,5]:
            print("& %3d,%3d "%(res[c,i,0],res[c,i,2]), end="")
        print("\\\\")

method_set = ['BIN','POI','PO-ORD-ACL','PO-VS-SL',
              'OH-BIN','OH-POI','OH-ORD-ACL','OH-VS-SL',
              'ORD-ACL','VS-SL','ACL','SL','AD']

for task in ['Z','A','S']:
    if task=='P':
        print("NLL")
    else:
        print("\nM%sE"%task)
    res = np.zeros((6,len(method_set),3))
    for c in [0,1,2,3,4,5]:
        size = size_set[c]
        for a in [0,1,2]:
            for b in [0,1,2,3,4,5,6,7]:
                if a!=2 or b<=4:
                    name = dataname_set[a][b]
                    ME = np.zeros((len(method_set),100))
                    for method in method_set:
                        if os.path.isfile("./Results/%s/%s/%s-%d.csv"%(method,task,name,size)):
                            data = np.loadtxt("./Results/%s/%s/%s-%d.csv"%(method,task,name,size),delimiter = ",")
                            if task=='P': ME[method_set.index(method),:] = data[:100,2]
                            if task=='Z': ME[method_set.index(method),:] = data[:100,3]
                            if task=='A': ME[method_set.index(method),:] = data[:100,7]
                            if task=='S': ME[method_set.index(method),:] = data[:100,11]
                    for i in range(len(method_set)):
                        for j in range(len(method_set)):
                            if i!=j:
                                test = stats.mannwhitneyu(ME[i,:],ME[j,:], alternative='less')
                                #i win, j lose
                                if   test.pvalue<=0.05:
                                    res[c,i,0]+=1
                                    res[c,j,2]+=1
                                else:
                                    res[c,i,1]+=1
                                    res[c,j,1]+=1
    for i in range(len(method_set)):
        for c in [0,1,2,3,4,5]:
            print("& %3d,%3d "%(res[c,i,0],res[c,i,2]), end="")
        print("\\\\")