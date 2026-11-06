import numpy as np
import torch

from torchhmc import thmc_trajectory_2dsk
from torchhmc import taction_2d

kappa=0.27
sklambda=0.022
tdev='cuda' if torch.cuda.is_available() else 'cpu'

print("#device=",tdev)


phi=torch.randn(64,4,4,device=tdev)

for i in range(1111):
  thmc_trajectory_2dsk(phi,0.05,kappa,sklambda)


aa=[]
ss=[]
pp=[]


oneovervol=1.0/( phi.shape[1]*phi.shape[2] )


for i in range(10000):
  thmc_trajectory_2dsk(phi,0.05,kappa,sklambda)
  action=taction_2d(phi,kappa,sklambda)
  sqa=torch.sum(phi*phi,(1,2))
  phia=torch.sum(phi,(1,2))
  for b in range(phi.shape[0]):
      aa.append(action[b].item())
      ss.append(sqa[b].item()*oneovervol)
      pp.append(phia[b].item()*oneovervol)


  
h1=np.histogram(aa,bins=200,density=True)
h2=np.histogram(ss,bins=200,density=True)
h3=np.histogram(pp,bins=200,density=True)

h1a=h1[0]
h1x=h1[1]
h2a=h2[0]
h2x=h2[1]
h3a=h3[0]
h3x=h3[1]

for i in range(len(h1x)-1):
  print(h1x[i],h1a[i],h2x[i],h2a[i],h3x[i],h3a[i])
