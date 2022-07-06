#import
import os
import numpy as np
import pandas as pd
from scipy import stats

size_set  = [25, 50, 100, 200, 400, 800]
datatype_set = ['DR5','DR10','OR','SY']
dataname_set = [['abalone-5','bank1-5','bank2-5','calhousing-5','census1-5','census2-5','computer1-5','computer2-5'], 
                ['abalone-10','bank1-10','bank2-10','calhousing-10','census1-10','census2-10','computer1-10','computer2-10'], 
                ['car','ERA','LEV','SWD','winequality-red']]

method_set = ['BIN','POI','PO-ORD-ACL','PO-VS-SL',
              'HET-BIN','HET-POI','HET-ORD-ACL','HET-VS-SL',
              'ACL','ORD-ACL','SL','VS-SL']

for task in ['P','Z','A','S']:
    if task=='P':
        print("NLL")
    else:
        print("\nM%sE"%task)
    ME1 = np.zeros((len(method_set),6))
    ME2 = np.zeros((len(method_set),6))
    ME3 = np.zeros((len(method_set),6))
    for a in [0,1,2]:
        for b in [0,1,2,3,4,5,6,7]:
            if a!=2 or b<=4:
                name = dataname_set[a][b]
                ME = np.zeros((len(method_set),6))
                for m in range(len(method_set)):
                    method = method_set[m]
                    for c in [0,1,2,3,4,5]:
                        size = size_set[c]
                        if os.path.isfile("./Results/%s/%s/%s-%d.csv"%(method,task,name,size)):
                            data = np.loadtxt("./Results/%s/%s/%s-%d.csv"%(method,task,name,size),delimiter = ",")
                            if task=='P': ME[m,c] = data[100,2]
                            if task=='Z': ME[m,c] = data[100,3]
                            if task=='A': ME[m,c] = data[100,7]
                            if task=='S': ME[m,c] = data[100,11]
                for c in [0,1,2,3,4,5]:
                    df = pd.DataFrame(ME[:,c])
                    for m in range(len(method_set)):
                        if df.rank()[0][m]<=1: ME1[m,c] += 1
                        if df.rank()[0][m]<=2: ME2[m,c] += 1
                        if df.rank()[0][m]<=3: ME3[m,c] += 1

    for m in range(len(method_set)):
        print("%s "%method_set[m],end="")
        for c in [0,1,2,3,4,5]:
            print("& %d,%d,%d "%(ME1[m,c],ME2[m,c],ME3[m,c]),end="")
        print("\\\\\n",end="")

