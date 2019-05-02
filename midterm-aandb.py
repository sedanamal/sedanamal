"""
Created on Thu May  2 07:08:50 2019

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import os

airfoils=[]
for root,dir,files in os.walk('.'):
    airfoils = np.append(airfoils,files)

selected = np.random.choice(airfoils,size = 100, replace = True, p =None)

for airfoil in selected:
        with open(airfoil) as f:
            lines = f.read().splitlines()
        test = any(c.isalpha() for c in lines[0])
        if test == True:
            with open(airfoil, 'r') as fin:
                data = fin.read().splitlines(True)
            with open(airfoil, 'w') as fout:
                fout.writelines(data[1:])

print(selected)
def normalization(x,y):
    while x[0]==1:
        if x[-1]==1:
            return(x,y)
        elif x[-1]!=1:
                maximum=np.max(x)
                for n in range (x):
                    xnew=[n/maximum for n in x]
                    ynew=[n/maximum for n in y]
                    n=n+1;
                    x=xnew
                    y=ynew
        return(x,y)

def plot (name,x,y):
    plt.ylim(ymax=0.2)
    plt.ylim(ymin=-0.2)
    plt.plot(x,y, label=name)
    plt.legend()
    
def chordline (x,y,LE,TE): 
    plt.plot([0,1], [LE, TE], ':', label='chordline')
    plt.legend()
    return((LE-TE)/(0-1))

def meancamberline(x,y,xup,yup,ylow):
    camberline = [(yup[i] + ylow[i])/2 for i in range(0,len(yup))]
    plt.plot(xup, camberline, ':', label='mean camber line')
    plt.legend()
    return(camberline)

def maximumthickness(airfoil,x,y,xup,yup,ylow):
    t = [yup[i] - ylow[i] for i in range(0,len(yup))]
    index = np.argmax(t)
    location = xup[index]
    plt.plot([location, location], [y[index], y[-index]], 'o-', label='maximum thickness')
    plt.legend()
    return(index)
    
for airfoil in selected:
    f = np.loadtxt(airfoil, dtype = float)
    xcoordinates = f.T[0]
    ycoordinates = f.T[1]
    
    nx, ny = normalization(xcoordinates,ycoordinates)
    
    plot(airfoil,nx,ny)
    
    minx = np.argmin(nx)
    xup = nx[:minx]
    xlow = nx[minx : -1]
    yup = ny[:minx]
    ylow = ny[minx : -1]
    
    le = ny[np.argmin(nx)]
    te = ny[np.argmax(nx)]
   
    chordline(nx,ny,le,te)
    
    meancamberline(nx,ny,xup,yup,ylow)
    
    maximumthickness(airfoil,nx,ny,xup,yup,ylow)
    
    plt.legend()
    plt.show()
    plt.savefig('airfoil')