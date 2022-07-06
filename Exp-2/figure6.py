#import
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams["font.size"] = 30
plt.rcParams["legend.markerscale"] = 1
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = 'k'
cmap = plt.get_cmap('jet')

size_set  = [25, 50, 100, 200, 400, 800]
datatype_set = ['DR5','DR10','OR','SY']
dataname_set = [['abalone-5','bank1-5','bank2-5','calhousing-5','census1-5','census2-5','computer1-5','computer2-5'], 
                ['abalone-10','bank1-10','bank2-10','calhousing-10','census1-10','census2-10','computer1-10','computer2-10'], 
                ['car','ERA','LEV','SWD','winequality-red']]


method_set = ['BIN','POI','PO-ORD-ACL','PO-VS-SL',
              'OH-BIN','OH-POI','OH-ORD-ACL','OH-VS-SL',
              'ACL','ORD-ACL','SL','VS-SL']
for task in ['P','Z','A','S']:
    for a in [0,1,2]:
        for b in [0,1,2,3,4,5,6,7]:
            if a!=2 or b<=4:
                name = dataname_set[a][b]
                fig=plt.figure(figsize=(18,9))
                ax = fig.add_subplot(111)
                plt.xscale('log')
                for method in method_set:
                    ME, ST = np.zeros(6), np.zeros(6)
                    for c in [0,1,2,3,4,5]:
                        size = size_set[c]
                        if os.path.isfile("./Results/%s/%s/%s-%d.csv"%(method,task,name,size)):
                            data = np.loadtxt("./Results/%s/%s/%s-%d.csv"%(method,task,name,size),delimiter = ",")
                            if task=='P': ME[c], ST[c] = data[100,2], data[101,2]
                            if task=='Z': ME[c], ST[c] = data[100,3], data[101,3]
                            if task=='A': ME[c], ST[c] = data[100,7], data[101,7]
                            if task=='S': ME[c], ST[c] = data[100,11], data[101,11]
                    plt.plot(np.array(size_set), ME, lw=1, color=cmap(method_set.index(method)/(len(method_set)-1)))
                    plt.errorbar(np.array(size_set), ME, yerr=ST, color=cmap(method_set.index(method)/(len(method_set)-1)), lw=1, capthick=1, label=method, capsize=10)
                plt.title("%s"%name)
                if task=='P':
                    plt.ylabel("NLL")
                else:
                    plt.ylabel("M%sE"%task)
                plt.xlabel(r"$n_{\rm tra}$")
                plt.xticks(size_set,["25", "50", "100", "200", "400", "800"])
                plt.legend(loc="upper right", fontsize=15)
                plt.grid()
                plt.tight_layout()
                if os.path.isdir("./Figures/%s"%task)==False: os.makedirs("./Figures/%s"%task)
                plt.savefig("./Figures/%s/%s-%s.png"%(task,name,task),bbox_inches="tight", pad_inches=0.01,facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close()
