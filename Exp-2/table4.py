#import
import os
import numpy as np
from scipy import stats

size_set  = [25, 50, 100, 200, 400, 800]
datatype_set = ['DR5','DR10','OR']
dataname_set = [['abalone-5','bank1-5','bank2-5','calhousing-5','census1-5','census2-5','computer1-5','computer2-5'], 
                ['abalone-10','bank1-10','bank2-10','calhousing-10','census1-10','census2-10','computer1-10','computer2-10'], 
                ['car','ERA','LEV','SWD','winequality-red']]

#'POI' vs. 'PO-ORD-ACL'
method_set = ['POI','PO-ORD-ACL']
print("\n",method_set)
for task in ['P','Z','A','S']:
    if task=='P':
        print("NLL ",end="")
    else:
        print("\nM%sE "%task,end="")
    for c in [0,1,2,3,4,5]:
        size = size_set[c]
        res = np.zeros(3)
        for a in [0,1,2]:
            for b in [0,1,2,3,4,5,6,7]:
                if a!=2 or b<=4:
                    name = dataname_set[a][b]
                    ME = np.zeros((2,100))
                    for method in method_set:
                        if os.path.isfile("./Results/%s/%s/%s-%d.csv"%(method,task,name,size)):
                            data = np.loadtxt("./Results/%s/%s/%s-%d.csv"%(method,task,name,size),delimiter = ",")
                            if task=='P': ME[method_set.index(method),:] = data[:100,2]
                            if task=='Z': ME[method_set.index(method),:] = data[:100,3]
                            if task=='A': ME[method_set.index(method),:] = data[:100,7]
                            if task=='S': ME[method_set.index(method),:] = data[:100,11]
                    test1 = stats.mannwhitneyu(ME[0,:],ME[1,:], alternative='less')
                    test2 = stats.mannwhitneyu(ME[0,:],ME[1,:], alternative='greater')
                    #i win, j lose
                    if   test1.pvalue<=0.05: res[0]+=1
                    elif test2.pvalue<=0.05: res[2]+=1
                    else: res[1]+=1
        print("& %d,%d "%(res[0],res[2]),end="")
    print("\\\\",end="")

#'BIN' vs. 'POI' vs. 'PO-ORD-ACL' vs. 'PO-VS-SL'
method_set = ['BIN','POI','PO-ORD-ACL','PO-VS-SL']
print("\n",method_set)
for task in ['P','Z','A','S']:
    if task=='P':
        print("NLL ",end="")
    else:
        print("\nM%sE "%task,end="")
    for c in [0,1,2,3,4,5]:
        size = size_set[c]
        res = np.zeros((4,3))
        for a in [0,1,2]:
            for b in [0,1,2,3,4,5,6,7]:
                if a!=2 or b<=4:
                    name = dataname_set[a][b]
                    ME = np.zeros((4,100))
                    for method in method_set:
                        if os.path.isfile("./Results/%s/%s/%s-%d.csv"%(method,task,name,size)):
                            data = np.loadtxt("./Results/%s/%s/%s-%d.csv"%(method,task,name,size),delimiter = ",")
                            if task=='P': ME[method_set.index(method),:] = data[:100,2]
                            if task=='Z': ME[method_set.index(method),:] = data[:100,3]
                            if task=='A': ME[method_set.index(method),:] = data[:100,7]
                            if task=='S': ME[method_set.index(method),:] = data[:100,11]
                    for i in range(4):
                        for j in range(4):
                            if i!=j:
                                test = stats.mannwhitneyu(ME[i,:],ME[j,:], alternative='less')
                                #i win, j lose
                                if   test.pvalue<=0.05:
                                    res[i,0]+=1
                                    res[j,2]+=1
                                else:
                                    res[i,1]+=1
                                    res[j,1]+=1
        print("& %d,%d,%d,%d "%(res[0,0],res[1,0],res[2,0],res[3,0]),end="")
    print("\\\\",end="")

#'BIN' vs. 'OH-BIN'
method_set = ['BIN','OH-BIN']
print("\n",method_set)
for task in ['P','Z','A','S']:
    if task=='P':
        print("NLL ",end="")
    else:
        print("\nM%sE "%task,end="")
    for c in [0,1,2,3,4,5]:
        size = size_set[c]
        res = np.zeros(3)
        for a in [0,1,2]:
            for b in [0,1,2,3,4,5,6,7]:
                if a!=2 or b<=4:
                    name = dataname_set[a][b]
                    ME = np.zeros((2,100))
                    for method in method_set:
                        if os.path.isfile("./Results/%s/%s/%s-%d.csv"%(method,task,name,size)):
                            data = np.loadtxt("./Results/%s/%s/%s-%d.csv"%(method,task,name,size),delimiter = ",")
                            if task=='P': ME[method_set.index(method),:] = data[:100,2]
                            if task=='Z': ME[method_set.index(method),:] = data[:100,3]
                            if task=='A': ME[method_set.index(method),:] = data[:100,7]
                            if task=='S': ME[method_set.index(method),:] = data[:100,11]
                    test1 = stats.mannwhitneyu(ME[0,:],ME[1,:], alternative='less')
                    test2 = stats.mannwhitneyu(ME[0,:],ME[1,:], alternative='greater')
                    #i win, j lose
                    if   test1.pvalue<=0.05: res[0]+=1
                    elif test2.pvalue<=0.05: res[2]+=1
                    else: res[1]+=1
        print("& %d,%d "%(res[0],res[2]),end="")
    print("\\\\",end="")

#'POI' vs. 'OH-POI'
method_set = ['POI','OH-POI']
print("\n",method_set)
for task in ['P','Z','A','S']:
    if task=='P':
        print("NLL ",end="")
    else:
        print("\nM%sE "%task,end="")
    for c in [0,1,2,3,4,5]:
        size = size_set[c]
        res = np.zeros(3)
        for a in [0,1,2]:
            for b in [0,1,2,3,4,5,6,7]:
                if a!=2 or b<=4:
                    name = dataname_set[a][b]
                    ME = np.zeros((2,100))
                    for method in method_set:
                        if os.path.isfile("./Results/%s/%s/%s-%d.csv"%(method,task,name,size)):
                            data = np.loadtxt("./Results/%s/%s/%s-%d.csv"%(method,task,name,size),delimiter = ",")
                            if task=='P': ME[method_set.index(method),:] = data[:100,2]
                            if task=='Z': ME[method_set.index(method),:] = data[:100,3]
                            if task=='A': ME[method_set.index(method),:] = data[:100,7]
                            if task=='S': ME[method_set.index(method),:] = data[:100,11]
                    test1 = stats.mannwhitneyu(ME[0,:],ME[1,:], alternative='less')
                    test2 = stats.mannwhitneyu(ME[0,:],ME[1,:], alternative='greater')
                    #i win, j lose
                    if   test1.pvalue<=0.05: res[0]+=1
                    elif test2.pvalue<=0.05: res[2]+=1
                    else: res[1]+=1
        print("& %d,%d "%(res[0],res[2]),end="")
    print("\\\\",end="")

#'OH-BIN' vs. 'OH-POI' vs. 'OH-ORD-ACL' vs. 'OH-VS-SL'
method_set = ['OH-BIN','OH-POI','OH-ORD-ACL','OH-VS-SL']
print("\n",method_set)
for task in ['P','Z','A','S']:
    if task=='P':
        print("NLL ",end="")
    else:
        print("\nM%sE "%task,end="")
    for c in [0,1,2,3,4,5]:
        size = size_set[c]
        res = np.zeros((4,3))
        for a in [0,1,2]:
            for b in [0,1,2,3,4,5,6,7]:
                if a!=2 or b<=4:
                    name = dataname_set[a][b]
                    ME = np.zeros((4,100))
                    for method in method_set:
                        if os.path.isfile("./Results/%s/%s/%s-%d.csv"%(method,task,name,size)):
                            data = np.loadtxt("./Results/%s/%s/%s-%d.csv"%(method,task,name,size),delimiter = ",")
                            if task=='P': ME[method_set.index(method),:] = data[:100,2]
                            if task=='Z': ME[method_set.index(method),:] = data[:100,3]
                            if task=='A': ME[method_set.index(method),:] = data[:100,7]
                            if task=='S': ME[method_set.index(method),:] = data[:100,11]
                    for i in range(4):
                        for j in range(4):
                            if i!=j:
                                test = stats.mannwhitneyu(ME[i,:],ME[j,:], alternative='less')
                                #i win, j lose
                                if   test.pvalue<=0.05:
                                    res[i,0]+=1
                                    res[j,2]+=1
                                else:
                                    res[i,1]+=1
                                    res[j,1]+=1
        print("& %d,%d,%d,%d "%(res[0,0],res[1,0],res[2,0],res[3,0]),end="")
    print("\\\\",end="")

#'PO-ORD-ACL' vs. 'OH-ORD-ACL' vs. 'ORD-ACL'
method_set = ['PO-ORD-ACL','OH-ORD-ACL','ORD-ACL']
print("\n",method_set)
for task in ['P','Z','A','S']:
    if task=='P':
        print("NLL ",end="")
    else:
        print("\nM%sE "%task,end="")
    for c in [0,1,2,3,4,5]:
        size = size_set[c]
        res = np.zeros((3,3))
        for a in [0,1,2]:
            for b in [0,1,2,3,4,5,6,7]:
                if a!=2 or b<=4:
                    name = dataname_set[a][b]
                    ME = np.zeros((3,100))
                    for method in method_set:
                        if os.path.isfile("./Results/%s/%s/%s-%d.csv"%(method,task,name,size)):
                            data = np.loadtxt("./Results/%s/%s/%s-%d.csv"%(method,task,name,size),delimiter = ",")
                            if task=='P': ME[method_set.index(method),:] = data[:100,2]
                            if task=='Z': ME[method_set.index(method),:] = data[:100,3]
                            if task=='A': ME[method_set.index(method),:] = data[:100,7]
                            if task=='S': ME[method_set.index(method),:] = data[:100,11]
                    for i in range(3):
                        for j in range(3):
                            if i!=j:
                                test = stats.mannwhitneyu(ME[i,:],ME[j,:], alternative='less')
                                #i win, j lose
                                if   test.pvalue<=0.05:
                                    res[i,0]+=1
                                    res[j,2]+=1
                                else:
                                    res[i,1]+=1
                                    res[j,1]+=1
        print("& %d,%d,%d "%(res[0,0],res[1,0],res[2,0]),end="")
    print("\\\\",end="")

#'PO-VS-SL' vs. 'OH-VS-SL' vs. 'VS-SL'
method_set = ['PO-VS-SL','OH-VS-SL','VS-SL']
print("\n",method_set)
for task in ['P','Z','A','S']:
    if task=='P':
        print("NLL ",end="")
    else:
        print("\nM%sE "%task,end="")
    for c in [0,1,2,3,4,5]:
        size = size_set[c]
        res = np.zeros((3,3))
        for a in [0,1,2]:
            for b in [0,1,2,3,4,5,6,7]:
                if a!=2 or b<=4:
                    name = dataname_set[a][b]
                    ME = np.zeros((3,100))
                    for method in method_set:
                        if os.path.isfile("./Results/%s/%s/%s-%d.csv"%(method,task,name,size)):
                            data = np.loadtxt("./Results/%s/%s/%s-%d.csv"%(method,task,name,size),delimiter = ",")
                            if task=='P': ME[method_set.index(method),:] = data[:100,2]
                            if task=='Z': ME[method_set.index(method),:] = data[:100,3]
                            if task=='A': ME[method_set.index(method),:] = data[:100,7]
                            if task=='S': ME[method_set.index(method),:] = data[:100,11]
                    for i in range(3):
                        for j in range(3):
                            if i!=j:
                                test = stats.mannwhitneyu(ME[i,:],ME[j,:], alternative='less')
                                #i win, j lose
                                if   test.pvalue<=0.05:
                                    res[i,0]+=1
                                    res[j,2]+=1
                                else:
                                    res[i,1]+=1
                                    res[j,1]+=1
        print("& %d,%d,%d "%(res[0,0],res[1,0],res[2,0]),end="")
    print("\\\\",end="")

#'ORD-ACL' vs. 'ACL'
method_set = ['ORD-ACL','ACL']
print("\n",method_set)
for task in ['P','Z','A','S']:
    if task=='P':
        print("NLL ",end="")
    else:
        print("\nM%sE "%task,end="")
    for c in [0,1,2,3,4,5]:
        size = size_set[c]
        res = np.zeros(3)
        for a in [0,1,2]:
            for b in [0,1,2,3,4,5,6,7]:
                if a!=2 or b<=4:
                    name = dataname_set[a][b]
                    ME = np.zeros((2,100))
                    for method in method_set:
                        if os.path.isfile("./Results/%s/%s/%s-%d.csv"%(method,task,name,size)):
                            data = np.loadtxt("./Results/%s/%s/%s-%d.csv"%(method,task,name,size),delimiter = ",")
                            if task=='P': ME[method_set.index(method),:] = data[:100,2]
                            if task=='Z': ME[method_set.index(method),:] = data[:100,3]
                            if task=='A': ME[method_set.index(method),:] = data[:100,7]
                            if task=='S': ME[method_set.index(method),:] = data[:100,11]
                    test1 = stats.mannwhitneyu(ME[0,:],ME[1,:], alternative='less')
                    test2 = stats.mannwhitneyu(ME[0,:],ME[1,:], alternative='greater')
                    #i win, j lose
                    if   test1.pvalue<=0.05: res[0]+=1
                    elif test2.pvalue<=0.05: res[2]+=1
                    else: res[1]+=1
        print("& %d,%d "%(res[0],res[2]),end="")
    print("\\\\",end="")

#'VS-SL' vs. 'SL'
method_set = ['VS-SL','SL']
print("\n",method_set)
for task in ['P','Z','A','S']:
    if task=='P':
        print("NLL ",end="")
    else:
        print("\nM%sE "%task,end="")
    for c in [0,1,2,3,4,5]:
        size = size_set[c]
        res = np.zeros(3)
        for a in [0,1,2]:
            for b in [0,1,2,3,4,5,6,7]:
                if a!=2 or b<=4:
                    name = dataname_set[a][b]
                    ME = np.zeros((2,100))
                    for method in method_set:
                        if os.path.isfile("./Results/%s/%s/%s-%d.csv"%(method,task,name,size)):
                            data = np.loadtxt("./Results/%s/%s/%s-%d.csv"%(method,task,name,size),delimiter = ",")
                            if task=='P': ME[method_set.index(method),:] = data[:100,2]
                            if task=='Z': ME[method_set.index(method),:] = data[:100,3]
                            if task=='A': ME[method_set.index(method),:] = data[:100,7]
                            if task=='S': ME[method_set.index(method),:] = data[:100,11]
                    test1 = stats.mannwhitneyu(ME[0,:],ME[1,:], alternative='less')
                    test2 = stats.mannwhitneyu(ME[0,:],ME[1,:], alternative='greater')
                    #i win, j lose
                    if   test1.pvalue<=0.05: res[0]+=1
                    elif test2.pvalue<=0.05: res[2]+=1
                    else: res[1]+=1
        print("& %d,%d "%(res[0],res[2]),end="")
    print("\\\\",end="")

#'ORD-ACL' vs. 'VS-SL' vs. 'ACL','SL'
method_set = ['ORD-ACL','VS-SL','ACL','SL']
print("\n",method_set)
for task in ['P','Z','A','S']:
    if task=='P':
        print("NLL ",end="")
    else:
        print("\nM%sE "%task,end="")
    for c in [0,1,2,3,4,5]:
        size = size_set[c]
        res = np.zeros((4,3))
        for a in [0,1,2]:
            for b in [0,1,2,3,4,5,6,7]:
                if a!=2 or b<=4:
                    name = dataname_set[a][b]
                    ME = np.zeros((4,100))
                    for method in method_set:
                        if os.path.isfile("./Results/%s/%s/%s-%d.csv"%(method,task,name,size)):
                            data = np.loadtxt("./Results/%s/%s/%s-%d.csv"%(method,task,name,size),delimiter = ",")
                            if task=='P': ME[method_set.index(method),:] = data[:100,2]
                            if task=='Z': ME[method_set.index(method),:] = data[:100,3]
                            if task=='A': ME[method_set.index(method),:] = data[:100,7]
                            if task=='S': ME[method_set.index(method),:] = data[:100,11]
                    for i in range(4):
                        for j in range(4):
                            if i!=j:
                                test = stats.mannwhitneyu(ME[i,:],ME[j,:], alternative='less')
                                #i win, j lose
                                if   test.pvalue<=0.05:
                                    res[i,0]+=1
                                    res[j,2]+=1
                                else:
                                    res[i,1]+=1
                                    res[j,1]+=1
        print("& %d,%d,%d,%d "%(res[0,0],res[1,0],res[2,0],res[3,0]),end="")
    print("\\\\",end="")
