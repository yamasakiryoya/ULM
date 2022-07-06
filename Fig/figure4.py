#coding: utf-8
import numpy as np
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit
from scipy.special import comb
plt.rcParams["font.size"] = 30
plt.rcParams["legend.markerscale"] = 1
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = 'k'
plt.rcParams.update({
    "text.usetex": True,
	"text.latex.preamble": r'\usepackage{bm}'})

cla=10
N=10000
fx=np.linspace(-1,10,N)
fig=plt.figure(figsize=(28,7))
cmap = plt.get_cmap('jet')

def CPD(u,k,s):
	b = np.array([0,1,2,3,4,5,6,7,8])
	b = np.concatenate((b,np.array([10000])))
	res = np.zeros(cla)
	for j in range(1,cla):
		res[j] = res[j-1]+(b[j]-u)
	arr = np.array([np.exp(-res[j]/s) for j in range(cla)])
	return arr[k]/np.sum(arr)

s=1
ax1 = fig.add_subplot(1,3,1)
arr = np.zeros((N,cla))
for k in range(cla):
	arr[:,k]=np.array([CPD(fx[i],k,s) for i in range(N)])
	ax1.plot(fx,arr[:,k],lw=3,linestyle="-",color=cmap(k/(cla-1)),label=r"$y=%s$"%(k+1))
ax1.set_xlim(-1,10)
ax1.set_xticks([0,3,6,9])
ax1.set_xticklabels([r'$0$',r'$3$',r'$6$',r'$9$'])
ax1.set_ylim(0.0,1.0)
ax1.set_xlabel(r'$u$')
ax1.set_ylabel(r'$(P_{\rm acl}(y;\bm{b}-u\cdot\bm{1}))_{y\in[%d]}$'%cla)
ax1.set_title(r'$\bm{b}=(0,1,2,3,4,5,6,7,8)$')
ax1.grid()

def CPD(u,k,s):
	b = np.array([0,0.5,1.5,2,3.5,4,4.5,7,8])
	b = np.concatenate((b,np.array([10000])))
	res = np.zeros(cla)
	for j in range(1,cla):
		res[j] = res[j-1]+(b[j]-u)
	arr = np.array([np.exp(-res[j]/s) for j in range(cla)])
	return arr[k]/np.sum(arr)

s=1
ax2 = fig.add_subplot(1,3,2)
arr = np.zeros((N,cla))
for k in range(cla):
	arr[:,k]=np.array([CPD(fx[i],k,s) for i in range(N)])
	ax2.plot(fx,arr[:,k],lw=3,linestyle="-",color=cmap(k/(cla-1)),label=r"$y=%s$"%(k+1))
ax2.set_xlim(-1,10)
ax2.set_xticks([0,3,6,9])
ax2.set_xticklabels([r'$0$',r'$3$',r'$6$',r'$9$'])
ax2.set_ylim(0.0,1.0)
ax2.axes.yaxis.set_ticklabels([])
ax2.set_xlabel(r'$u$')
ax2.set_title(r'$\bm{b}=(0,1,1.5,2,3.5,4,4.5,7,8)$')
ax2.grid()

def CPD(u,k,s):
	b = np.array([0,0.5,2,1.5,3.5,4,7,4.5,8])
	b = np.concatenate((b,np.array([10000])))
	res = np.zeros(cla)
	for j in range(1,cla):
		res[j] = res[j-1]+(b[j]-u)
	arr = np.array([np.exp(-res[j]/s) for j in range(cla)])
	return arr[k]/np.sum(arr)

s=1
ax3 = fig.add_subplot(1,3,3)
arr = np.zeros((N,cla))
for k in range(cla):
	arr[:,k]=np.array([CPD(fx[i],k,s) for i in range(N)])
	ax3.plot(fx,arr[:,k],lw=3,linestyle="-",color=cmap(k/(cla-1)),label=r"$y=%s$"%(k+1))
ax3.set_xlim(-1,10)
ax3.set_xticks([0,3,6,9])
ax3.set_xticklabels([r'$0$',r'$3$',r'$6$',r'$9$'])
ax3.set_ylim(0.0,1.0)
ax3.axes.yaxis.set_ticklabels([])
ax3.set_xlabel(r'$u$')
ax3.set_title(r'$\bm{b}=(0,1,2,1.5,3.5,4,7,4.5,8)$')
ax3.grid()

cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), aspect=35, pad=0.015, orientation='vertical')
cbar.set_ticks([0,1/9,2/9,3/9,4/9,5/9,6/9,7/9,8/9,1]); cbar.set_ticklabels([r"$1$",r"$2$",r"$3$",r"$4$",r"$5$",r"$6$",r"$7$",r"$8$",r"$9$",r"$10$"]); cbar.set_label(r"$y$")
plt.tight_layout()
plt.savefig("./PO-ACL_%d.png"%cla,bbox_inches="tight", pad_inches=0.01,facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
