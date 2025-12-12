# %% [markdown]
# Our Lagrangian is
# 
# $$\mathcal{L} = \frac{1}{2}\tilde{m}\tilde{L}\left(\tilde{\dot\theta}^2 + \tilde{A}^2\sin^2(\tau)\sin^2(\tilde{\theta})\right) + \tilde{L}\tilde{\dot\theta}^2 \sin^2(\tilde\theta) + \cos(\tilde\theta)(\tilde{m}+ 1).$$
# 
# The Euler--Lagrange equation of motion is therefore, on the left side,
# $$\frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \tilde{\dot\theta}}\right) = \frac{d}{dt}\left(\tilde{m} \tilde{L} \tilde{\dot\theta} + 2\tilde{L} \tilde{\dot\theta}\sin^2(\tilde\theta)\right) = \tilde{m}\tilde{L} \tilde{\ddot\theta} + 4\tilde{L}\tilde{\dot\theta}^2\sin(\tilde\theta)\cos(\tilde\theta) + 2\tilde{L}\tilde{\ddot\theta}\sin^2(\tilde\theta).$$
# 
# On the right side, we have $$\frac{\partial \mathcal{L}}{\partial \tilde \theta} = \tilde{m}\tilde{L} \tilde{A}^2 \sin^2(\tau) \sin(\tilde\theta)\cos(\tilde\theta) + 2\tilde{L} \tilde{\dot\theta}^2 \sin(\tilde\theta)\cos(\tilde\theta) - \sin(\tilde\theta)(\tilde{m} + 1).$$
# 
# So, the E--L equation is $$\tilde{m}\tilde{L} \tilde{\ddot\theta} + 4\tilde{L}\tilde{\dot\theta}^2\sin(\tilde\theta)\cos(\tilde\theta) + 2\tilde{L}\tilde{\ddot\theta}\sin^2(\tilde\theta) = \tilde{m}\tilde{L} \tilde{A}^2 \sin^2(\tau) \sin(\tilde\theta)\cos(\tilde\theta) + 2\tilde{L} \tilde{\dot\theta}^2 \sin(\tilde\theta)\cos(\tilde\theta) - \sin(\tilde\theta)(\tilde{m} + 1).$$
# 

# %% [markdown]
# We can rewrite this as $$\tilde{\ddot\theta}= \frac{\tilde{m}\tilde{L} \tilde{A}^2 \sin^2(\tau) \sin(\tilde\theta)\cos(\tilde\theta) + 2\tilde{L} \tilde{\dot\theta}^2 \sin(\tilde\theta)\cos(\tilde\theta) - \sin(\tilde\theta)(\tilde{m} + 1)  - 4\tilde{L}\tilde{\dot\theta}^2\sin(\tilde\theta)\cos(\tilde\theta)}{\tilde{m}\tilde{L}  +  2\tilde{L}\sin^2(\tilde\theta) }$$

# %%
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from numpy import sin, cos
import sys
from multiprocessing import Pool

# %% [markdown]
# We can rewrite this as $$\tilde{\ddot\theta}= \frac{\tilde{m}\tilde{L} \tilde{A}^2 \sin^2(\tau) \sin(\tilde\theta)\cos(\tilde\theta) + 2\tilde{L} \tilde{\dot\theta}^2 \sin(\tilde\theta)\cos(\tilde\theta) - \sin(\tilde\theta)(\tilde{m} + 1)  - 4\tilde{L}\tilde{\dot\theta}^2\sin(\tilde\theta)\cos(\tilde\theta)}{\tilde{m}\tilde{L}  +  2\tilde{L}\sin^2(\tilde\theta) }$$

# %% [markdown]
# ## We can now define our function

# %%
def ELE(tau, X, L, A, m):
    '''
        Equation of motion for system
        return:[ddot theta, dot theta]
     '''
    dtheta, theta = X
    ddtheta = (m * L * A**2 * sin(tau)**2 * sin(theta) * cos(theta) 
               + 2 * L * dtheta**2 * sin(theta) * cos(theta)
               - sin(theta) * (m + 1)
               - 4 * L * dtheta**2 * sin(theta) * cos(theta)
                ) / ( m * L + 2 * L * sin(theta)**2)

    return [ddtheta, dtheta]

# %%
def plotIC(thetaZero, dthetaZero, tau_max, L = 1, A = 1, m = 1, max_step = 1e-2, tauSlice = 0):
    timePhase = np.arange(0, tau_max // (2*np.pi)) * np.pi * 2 + tauSlice #np.linspace(0, tau_max, 1000)
    # print(timePhase)
    timePos = np.linspace(0, tau_max//2, 1000)
    # print(time)
    solution = solve_ivp(ELE, (0, tau_max), (dthetaZero, thetaZero), args = (L, A, m), dense_output=True, max_step=max_step)
    dthetasPhase, thetasPhase = solution.sol(timePhase)
    dthetasPos, thetasPos = solution.sol(timePos)

    thetasPos = np.mod(thetasPos + np.pi, 2 * np.pi) - np.pi
    thetasPhase= np.mod(thetasPhase + np.pi, 2 * np.pi) - np.pi


    fig, ax = plt.subplots(1,2, figsize=(8,4), tight_layout=True)
    ax[0].plot(timePos, thetasPos)
    ax[0].set_xlabel(r'$\tau$ (dimensionless time)')
    ax[0].set_ylabel(r'$\tilde\theta$ ("dimensionless" angle)')
    ax[1].scatter(thetasPhase, dthetasPhase, s=1)
    ax[1].set_xlabel(r'$\tilde\theta$')
    ax[1].set_ylabel(r'$\dot{\tilde{\theta}}$')
    ax[1].set_xlim(xmin=-1*(np.pi/2 + 0.1), xmax=np.pi/2 + 0.1)
    # ax[1].set_xlim(xmin=-0.1, xmax=np.pi*2 + 0.1)

    ax[1].yaxis.set_label_position('right')

    fig.suptitle(rf"$\theta$ over time for $\tilde m = {m:.2f}, \tilde A = {A:.2f}, \tilde L ={L:.2f}$ and $\theta(0) = {thetaZero:.2f}, \dot\theta(0) = {dthetaZero:.2f}$")
    fig.savefig(f'Base.pdf')
    plt.close('all')



# %%


def plotDtheta(thetaZero, min,max, tau_max, L = 1, A = 1, m = 1, max_step = 1e-2, tauSlice = 0):
    dthetaList = np.linspace(min,max, 8)
    timePhase = np.arange(0, tau_max // (2*np.pi)) * np.pi * 2 + tauSlice #np.linspace(0, tau_max, 1000)
    numPoints = timePhase.size
    print(numPoints)
    #dtPlotPoints = np.zeros((numPoints))
    #tPlotPoints = np.zeros((numPoints))
    dtPlotPoints = []
    tPlotPoints = []
    for i,dt in enumerate(dthetaList): 
        print(i)
    # print(time)
        solution = solve_ivp(ELE, (0, tau_max), (dt, thetaZero), args = (L, A, m), dense_output=True, max_step=max_step)
        dthetasPhase, thetasPhase = solution.sol(timePhase)
        thetasPhase= np.mod(thetasPhase + np.pi, 2 * np.pi) - np.pi
        dtPlotPoints.append(dthetasPhase)
        tPlotPoints.append(thetasPhase)

    fig, ax = plt.subplots()
    for i in range(dthetaList.size):
        ax.scatter(tPlotPoints[i], dtPlotPoints[i], s=2, label =fr'$\dot{{\tilde{{\theta}}}} = $ {dthetaList[i]:.1f}')
        ax.set_xlabel(r'$\tilde\theta$')
        ax.set_ylabel(r'$\dot{\tilde{\theta}}$')
    ax.legend()
    fig.savefig(f'dtSurfaceSection.pdf')
    plt.close('all')


def plottheta(min,max, dthetaZero, tau_max, L = 1, A = 1, m = 1, max_step = 1e-2, tauSlice = 0):
    thetaList = np.linspace(min,max, 15)
    timePhase = np.arange(0, tau_max // (2*np.pi)) * np.pi * 2 + tauSlice #np.linspace(0, tau_max, 1000)
    numPoints = timePhase.size
    print(numPoints)
    #dtPlotPoints = np.zeros((numPoints))
    #tPlotPoints = np.zeros((numPoints))
    dtPlotPoints = []
    tPlotPoints = []
    for i,theta in enumerate(thetaList): 
        print(i)
    # print(time)
        solution = solve_ivp(ELE, (0, tau_max), (dthetaZero , theta), args = (L, A, m), dense_output=True, max_step=max_step)
        dthetasPhase, thetasPhase = solution.sol(timePhase)
        thetasPhase= np.mod(thetasPhase + np.pi, 2 * np.pi) - np.pi
        dtPlotPoints.append(dthetasPhase)
        tPlotPoints.append(thetasPhase)

    fig, ax = plt.subplots()
    for i in range(thetaList.size):
        ax.scatter(tPlotPoints[i], dtPlotPoints[i], s=2, label =fr'${{\tilde{{\theta}}}} = $ {thetaList[i]:.1f}')
        ax.set_xlabel(r'$\tilde\theta$')
        ax.set_ylabel(r'$\dot{\tilde{\theta}}$')
    # ax.legend()
    fig.savefig(f'thetaSurfaceSection.pdf')
    plt.close('all')


def plotAtheta(thetaZero, dthetaZero, min,max,tau_max, L = 1, m = 1, max_step = 1e-2, tauSlice = 0):
    A = np.linspace(min,max, 3) 
    A = np.concat([A,[5]])
    timePhase = np.arange(0, tau_max // (2*np.pi)) * np.pi * 2 + tauSlice #np.linspace(0, tau_max, 1000)
    numPoints = timePhase.size
    print(numPoints)
    #dtPlotPoints = np.zeros((numPoints))
    #tPlotPoints = np.zeros((numPoints))
    dtPlotPoints = []
    tPlotPoints = []
    for i,aa in enumerate(A): 
        print(i)
    # print(time)
        solution = solve_ivp(ELE, (0, tau_max), (dthetaZero , thetaZero), args = (L,aa, m), dense_output=True, max_step=max_step)
        dthetasPhase, thetasPhase = solution.sol(timePhase)
        thetasPhase= np.mod(thetasPhase + np.pi, 2 * np.pi) - np.pi
        dtPlotPoints.append(dthetasPhase)
        tPlotPoints.append(thetasPhase)

    fig, ax = plt.subplots()
    for i in range(A.size):
        ax.scatter(tPlotPoints[i], dtPlotPoints[i], s=2, label =fr'$A= $ {A[i]:.1f}')
        ax.set_xlabel(r'$\tilde\theta$')
        ax.set_ylabel(r'$\dot{\tilde{\theta}}$')
    ax.legend()
    fig.savefig(f'AaaSurfaceOfSections.pdf')
    plt.close('all')


# Tlist = np.linspace(np.pi/4 - 0.5,np.pi/4 + 0.5, 10)
ICList = np.linspace(0.0,3, 50)
# TL, LL = np.meshgrid(Tlist, Llist)
# TL = np.ravel(TL)
# LL = np.ravel(LL)
# IC = np.zeros((TL.size,2))
# IC[:,0] = LL
# IC[:,1] = TL

# def helper(IC):
#     plotIC(IC[1], 0, 1000, L=IC[0], A=1)
def helper(IC):
    plotIC(np.pi/4, IC, 10, L=1, A=1)

# for i in range(len(AL)):
#     print(i)
#     plotIC(np.pi/4, 0, 1000, L=LL[i], A=AL[i])

# if __name__ == "__main__":
#     with Pool(processes=5) as pool:
#         features = pool.map(helper,ICList)
#         pool.close()
#         pool.join()

# %%
# plotIC(0.0015, 0, 5000, L=2, A=2, tauSlice = 0)

# plotIC(np.pi/4, 0, 1000, L=1, A=1)

plotAtheta(np.pi/4, 0, 0.2 ,6, 2000)


# %%
