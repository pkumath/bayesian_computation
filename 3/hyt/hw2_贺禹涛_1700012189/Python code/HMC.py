import sys
sys.path.append('/raid/heyutao/code/hw/ITE/')

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import invgamma
from numpy import random
from pandas import DataFrame
import pandas as pd
from ite.cost.x_analytical_values import analytical_value_d_kullback_leibler
from ite.cost.x_factory import co_factory

sth,sy=1,5

def U(theta,y):
  return sum((y-theta[0]-theta[1]**2)**2)/(2*sy**2)+sum(theta**2)/(2*sth**2)
  
def K(r):
  return sum(r**2)/2
  
def dUd1(theta,y):
  #print(sy,sth)
  #print(sum(-y+theta[0]+theta[1]**2)/sy**2)
  ans=sum(-y+theta[0]+theta[1]**2)/sy**2+theta[0]/sth**2
  #print(ans)
  return ans  
  
def dUd2(theta,y):
  ans=2*theta[1]*sum(-y+theta[0]+theta[1]**2)/sy**2+theta[1]/sth**2
  #print(ans)
  return ans
  

def leap_frog(y,theta,r,eps):
  r1_half=r[0]-eps/2*dUd1(theta,y)
  r2_half=r[1]-eps/2*dUd2(theta,y)
  th1=theta[0]+eps*r1_half
  th2=theta[1]+eps*r2_half
  r1=r1_half-eps/2*dUd1([th1,th2],y)
  r2=r2_half-eps/2*dUd2([th1,th2],y)
  #print([th1,th2])
  return np.array([th1,th2]),np.array([r1,r2])
  
y=np.random.normal(1,sy,10000)
theta=np.random.normal(0,sth,2)

T=100000
r=np.random.normal(0,1,2)
param_rec=np.zeros([T,2])
for t in range(T):
  oldH=U(theta,y)+K(r)
  new_theta=theta*1
  new_r=r*1
  L=np.random.randint(1,30)
  for i in range(L):
    new_theta,new_r=leap_frog(y,new_theta*1,new_r*1,0.01)
    #print(new_theta)
  new_r=-new_r
  newH=U(new_theta,y)+K(new_r)
  p=min(1,np.exp(-newH+oldH))
  #print(p)
  if(np.random.uniform(0,1)<p):
    theta=new_theta
  #print(t,': ',theta)
  r=np.random.normal(0,1,2)
  param_rec[t,:]=theta

np.savetxt('/raid/heyutao/code/hw/HMC.txt',param_rec[50000:T,:]) 

plt.scatter(param_rec[50000:T,0],param_rec[50000:T,1])
plt.xlabel('theta_1')
plt.ylabel('theta_2')
plt.grid(True)

plt.show()

