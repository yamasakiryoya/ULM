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

loc=5.4
cla=10
N=10000
fx=np.linspace(-1,10,N)
fig=plt.figure(figsize=(28,7))
cmap = plt.get_cmap('jet')

ax1 = fig.add_subplot(1,3,1)
arr=np.array([.001,.002,.003,.004,.74,.24,.004,.003,.002,.001])
for k in range(cla):
	ax1.plot(np.linspace(0.5+k,1.5+k,10), arr[k]*np.ones(10),lw=3,linestyle="-",color="k")
	if k<4:
		ax1.plot(np.linspace(0.5+k,1.5+k,10), arr[k]/arr[k+1]*np.ones(10),lw=3,linestyle="--",color="r")
	if k>4:
		ax1.plot(np.linspace(0.5+k,1.5+k,10), arr[k]/arr[k-1]*np.ones(10),lw=3,linestyle="--",color="r")
for k in range(cla-1):
	ax1.plot((1.5+k)*np.ones(10), np.linspace(arr[k],arr[k+1],10),lw=3,linestyle="-",color="k")
	if k<3:
		ax1.plot((1.5+k)*np.ones(10), np.linspace(arr[k]/arr[k+1],arr[k+1]/arr[k+2],10),lw=3,linestyle="--",color="r")
	if k>4:
		ax1.plot((1.5+k)*np.ones(10), np.linspace(arr[k]/arr[k-1],arr[k+1]/arr[k],10),lw=3,linestyle="--",color="r")
#
for k in range(cla+1):
	ax1.plot((0.5+k)*np.ones(10), np.linspace(0,1,10),lw=0.5,linestyle="-",color="gray")
ax1.plot(np.linspace(0,11,10), 0.2*np.ones(10), lw=0.5,linestyle="-",color="gray")
ax1.plot(np.linspace(0,11,10), 0.4*np.ones(10), lw=0.5,linestyle="-",color="gray")
ax1.plot(np.linspace(0,11,10), 0.6*np.ones(10), lw=0.5,linestyle="-",color="gray")
ax1.plot(np.linspace(0,11,10), 0.8*np.ones(10), lw=0.5,linestyle="-",color="gray")

ax1.set_xlim(0,cla+1)
ax1.set_xticks(range(1,11))
ax1.axes.xaxis.set_ticklabels([r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$',r'$7$',r'$8$',r'$9$',r'$10$'])
ax1.axes.yaxis.set_ticklabels([])
ax1.set_ylim(0.0,1.0)
ax1.set_xlabel(r'$k$')
ax1.set_title(r'$\sum_{k=1}^4p_k\approx%.2f, \sum_{k=6}^{10}p_k\approx%.2f$'%(np.sum(arr[:4]),np.sum(arr[5:])))
ax1.set_ylabel(r'$(p_k)_{k\in[10]}, \{\frac{p_k}{p_{k+1}}\}_{k=1}^4, \{\frac{p_k}{p_{k-1}}\}_{k=6}^{10}$')

ax2 = fig.add_subplot(1,3,2)
arr=np.array([.01,.01,.06,.21,.41,.21,.06,.02,.01,.0])
for k in range(cla):
	ax2.plot(np.linspace(0.5+k,1.5+k,10), arr[k]*np.ones(10),lw=3,linestyle="-",color="k")
	if k<4:
		ax2.plot(np.linspace(0.5+k,1.5+k,10), arr[k]/arr[k+1]*np.ones(10),lw=3,linestyle="--",color="r")
	if k>4:
		ax2.plot(np.linspace(0.5+k,1.5+k,10), arr[k]/arr[k-1]*np.ones(10),lw=3,linestyle="--",color="r")
for k in range(cla-1):
	ax2.plot((1.5+k)*np.ones(10), np.linspace(arr[k],arr[k+1],10),lw=3,linestyle="-",color="k")
	if k<3:
		ax2.plot((1.5+k)*np.ones(10), np.linspace(arr[k]/arr[k+1],arr[k+1]/arr[k+2],10),lw=3,linestyle="--",color="r")
	if k>4:
		ax2.plot((1.5+k)*np.ones(10), np.linspace(arr[k]/arr[k-1],arr[k+1]/arr[k],10),lw=3,linestyle="--",color="r")
#
for k in range(cla+1):
	ax2.plot((0.5+k)*np.ones(10), np.linspace(0,1,10),lw=0.5,linestyle="-",color="gray")
ax2.plot(np.linspace(0,11,10), 0.2*np.ones(10), lw=0.5,linestyle="-",color="gray")
ax2.plot(np.linspace(0,11,10), 0.4*np.ones(10), lw=0.5,linestyle="-",color="gray")
ax2.plot(np.linspace(0,11,10), 0.6*np.ones(10), lw=0.5,linestyle="-",color="gray")
ax2.plot(np.linspace(0,11,10), 0.8*np.ones(10), lw=0.5,linestyle="-",color="gray")

ax2.set_xlim(0,cla+1)
ax2.set_xticks(range(1,11))
ax2.axes.xaxis.set_ticklabels([r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$',r'$7$',r'$8$',r'$9$',r'$10$'])
ax2.axes.yaxis.set_ticklabels([])
ax2.set_ylim(0.0,1.0)
ax2.set_xlabel(r'$k$')
ax2.set_title(r'$\sum_{k=1}^4p_k\approx%.2f, \sum_{k=6}^{10}p_k\approx%.2f$'%(np.sum(arr[:4]),np.sum(arr[5:])))

ax3 = fig.add_subplot(1,3,3)
arr=np.array([.01,.02,.05,.1,.22,.18,.15,.12,.09,.06])
for k in range(cla):
	ax3.plot(np.linspace(0.5+k,1.5+k,10), arr[k]*np.ones(10),lw=3,linestyle="-",color="k")
	if k<4:
		ax3.plot(np.linspace(0.5+k,1.5+k,10), arr[k]/arr[k+1]*np.ones(10),lw=3,linestyle="--",color="r")
	if k>4:
		ax3.plot(np.linspace(0.5+k,1.5+k,10), arr[k]/arr[k-1]*np.ones(10),lw=3,linestyle="--",color="r")
for k in range(cla-1):
	ax3.plot((1.5+k)*np.ones(10), np.linspace(arr[k],arr[k+1],10),lw=3,linestyle="-",color="k")
	if k<3:
		ax3.plot((1.5+k)*np.ones(10), np.linspace(arr[k]/arr[k+1],arr[k+1]/arr[k+2],10),lw=3,linestyle="--",color="r")
	if k>4:
		ax3.plot((1.5+k)*np.ones(10), np.linspace(arr[k]/arr[k-1],arr[k+1]/arr[k],10),lw=3,linestyle="--",color="r")
#
for k in range(cla+1):
	ax3.plot((0.5+k)*np.ones(10), np.linspace(0,1,10),lw=0.5,linestyle="-",color="gray")
ax3.plot(np.linspace(0,11,10), 0.2*np.ones(10), lw=0.5,linestyle="-",color="gray")
ax3.plot(np.linspace(0,11,10), 0.4*np.ones(10), lw=0.5,linestyle="-",color="gray")
ax3.plot(np.linspace(0,11,10), 0.6*np.ones(10), lw=0.5,linestyle="-",color="gray")
ax3.plot(np.linspace(0,11,10), 0.8*np.ones(10), lw=0.5,linestyle="-",color="gray")

ax3.set_xlim(0,cla+1)
ax3.set_xticks(range(1,11))
ax3.axes.xaxis.set_ticklabels([r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$',r'$7$',r'$8$',r'$9$',r'$10$'])
ax3.axes.yaxis.set_ticklabels([])
ax3.set_ylim(0.0,1.0)
ax3.set_xlabel(r'$k$')
ax3.set_title(r'$\sum_{k=1}^4p_k\approx%.2f, \sum_{k=6}^{10}p_k\approx%.2f$'%(np.sum(arr[:4]),np.sum(arr[5:])))

plt.tight_layout()
plt.savefig("./scale.png",bbox_inches="tight", pad_inches=0.01,facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
