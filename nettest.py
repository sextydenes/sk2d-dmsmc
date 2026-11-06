#!/usr/bin/python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

import torchhmc
from torchhmc import thmc_trajectory_2dsk
from torchhmc import taction_2d
import torch.nn.functional as F

import argparse

parser=argparse.ArgumentParser(description='')

import config

parser.add_argument("-Nx",type=int,help="Nx",default=4)
parser.add_argument("-Ny",type=int,help="Ny",default=4)
parser.add_argument("-Nsamp",type=int,help="how many configs for training",default=2000)
parser.add_argument("-Nlearn",type=int,help="how many training steps",default=1000)
parser.add_argument("-load",type=str,help="load the state of the nn")
parser.add_argument("-whichnet",type=int,help="what kind of nn",default=1)
parser.add_argument("-Nfin",type=int,help="Nfin, final net width",default=1024)

args=config.add_defargs(parser)


#args=parser.parse_args()

#explsigma=args.explsigma
#nfile=args.nfile
#if nfile==None: nfile=1





sklambda=config.sklambda
kappa=config.kappa
explsigma=config.explsigma
link=config.link
nfile=config.nfile
widthfakt=config.widthfakt

print("config:",nfile,link,explsigma,kappa,sklambda)


gintfakt=1.0/(2.0*(link+np.log(explsigma)))
tdev='cuda' if torch.cuda.is_available() else 'cpu'

if args.dev!=None:
  tdev=args.dev



def goft(t):
  return np.power(explsigma,t)

def gint(t):
  return (np.power(explsigma,2.0*t)-np.exp(-2.0*link*t))/(2.0*(link+np.log(explsigma)))

def gint_nolinforce(t):
  return 0.5/np.log(explsigma)*(np.power(explsigma,2.0*t)-1.0)


def tgint(t):
  return (torch.pow(explsigma,2.0*t)-torch.exp(-2.0*link*t))*gintfakt

def sqrttgint(t):
  return torch.sqrt((torch.pow(explsigma,2.0*t)-torch.exp(-2.0*link*t))*gintfakt)

#/(2.0*(link+np.log(explsigma)))




def create_traindata_numpy(Nbig,Nx,Ny):
  torch_phidat = torch.empty(Nbig, Nx, Ny)
  torch_noiseamp = torch.empty(Nbig)
  torch_times = torch.empty(Nbig)
  phi=np.random.randn(Nx,Ny)
  for i in range(1111):
    hmc_trajectory(phi,0.02)
  for i in range(Nbig):
    for j in range(10):
      hmc_trajectory(phi,0.02)
    time=np.random.rand()
    torch_noiseamp[i]=np.sqrt(gint(time))
    torch_times[i]=time
    tt=torch.tensor(phi)
    torch_phidat[i]=tt
  return torch_phidat, torch_noiseamp , torch_times


def create_traindata(Nbig,Nx,Ny):
  torch_phidat = torch.randn(Nbig, Nx, Ny,device=tdev)
  torch_noiseamp = torch.empty(Nbig)
  torch_times = torch.empty(Nbig)
  for i in range(1111):
    thmc_trajectory_2dsk(torch_phidat,0.02,kappa,sklambda)
  for i in range(Nbig):
    time=np.random.rand()
    torch_noiseamp[i]=np.sqrt(gint(time))
    torch_times[i]=time
  torch_noiseamp=torch_noiseamp.to(tdev)
  torch_times=torch_times.to(tdev)
#  print("de sikerult",torch_noiseamp.device,tdev)
  
  return torch_phidat, torch_noiseamp , torch_times




#    sigsq=gint(time)
#    noise=np.random.randn(Nx,Ny)
#    noised=phi+noise*np.sqrt(sigsq)
#    tt=torch.tensor(noised)
#    torch_phidat[i]=tt
    
from nns import *




from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

def train(model, x_fields, t_scalars, times_scalars, loss_fn, epochs=10, batch_size=32, lr=1e-3):
    """
    model: instance of RNet
    x_fields: Tensor (Nlarge, Nx, Ny) - input 2D fields
    t_scalars: Tensor (Nlarge,) - input scalar times
    loss_fn: function(gradients, x_fields, t_scalars) -> scalar loss
    epochs: number of training epochs
    batch_size: minibatch size
    lr: learning rate
    """
    dataset = TensorDataset(x_fields, t_scalars, times_scalars)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    Nl=200
    lossbin=1.0/Nl
    hits=np.zeros(Nl)
    losses=np.zeros(Nl)
    
    
    
    for epoch in range(epochs):
        total_loss = 0.0
        if epoch % 1000 ==0:
          nnet.save('tnetstate.'+str(int(epoch/1000))+"."+str(nfile))
        for x_batch, t_batch , times_batch in dataloader:
            x_batch = x_batch.requires_grad_(True)  # Needed for jacrev gradients
            optimizer.zero_grad()

 #           print(x_batch.shape,t_batch.shape)

            noise = torch.randn(x_batch.shape,device=tdev)
            Nb=x_batch.shape[0]   
            times_b2=torch.rand(Nb,device=tdev)
            t_b2=sqrttgint(times_b2)
            
            Nbatch=x_batch.shape[0]
            noised = x_batch-noise*t_b2.view(Nbatch,1,1)            
            R, grad = model(noised, t_b2)

            loss=F.mse_loss(noise, grad*t_b2.view(Nbatch,1,1), reduction='mean') 
            #loss_fn(grad, x_batch, t_batch)

            loss.backward()
            optimizer.step()
            
            if epoch>epochs*0.99:
              for i in range(times_batch.shape[0]):
                 time=times_b2[i].item()
                 kisloss=F.mse_loss(noise[i],grad[i]*t_b2[i],reduction='mean')
                 mycol=int(time/lossbin)
                 hits[mycol]+=1
                 losses[mycol]+=kisloss.item()
            

            total_loss += loss.item() * x_batch.size(0)  # sum over batch

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")


    with open('diffa', 'w') as f:
      for i in range(Nl):
        print(i,i*lossbin,losses[i]/hits[i],file=f)

Nx=4
Ny=4
Nb=2        

Nx=args.Nx
Ny=args.Ny
Nb=args.Nb


#create data
#phidat,tdat,timesdat = create_traindata(args.Nsamp ,Nx,Ny)


torch.set_default_dtype(torch.float64)



if args.whichnet==1: nnet=RNet(args.Nx,args.Ny,args.Nx*args.Ny*widthfakt)
if args.whichnet==2: nnet=CNet1(args.Nx,args.Ny,3,widthfakt,1)
if args.whichnet==3: nnet=CNet2(args.Nx,args.Ny,3,widthfakt,2,32)
if args.whichnet==4: nnet=CNet3(args.Nx,args.Ny,3,widthfakt,4*widthfakt,32)
if args.whichnet==5: nnet=CNet4(args.Nx,args.Ny,3,widthfakt,4*widthfakt,32)
if args.whichnet==6: nnet=CNet5(args.Nx,args.Ny,3,widthfakt,4*widthfakt,32)
if args.whichnet==7: nnet=CNet6(args.Nx,args.Ny,3,widthfakt,4*widthfakt,256)
if args.whichnet==8: nnet=CNet7(args.Nx,args.Ny,3,widthfakt,4*widthfakt,256)
if args.whichnet==9: nnet=CNet8(args.Nx,args.Ny,3,widthfakt,4*widthfakt,args.Nfin,kappa=config.kappa,sklambda=config.sklambda)

if args.whichnet==60: nnet=CNet60(args.Nx,args.Ny,3,widthfakt,4*widthfakt,args.Nfin,gint(1.0),kappa=config.kappa,sklambda=config.sklambda)
if args.whichnet==61: nnet=CNet61(args.Nx,args.Ny,3,widthfakt,4*widthfakt,args.Nfin,kappa=config.kappa,sklambda=config.sklambda)
if args.whichnet==62: nnet=CNet62(args.Nx,args.Ny,3,widthfakt,2,32)
if args.whichnet==63: nnet=CNet63(args.Nx,args.Ny,3,widthfakt,2,32)
if args.whichnet==64: nnet=CNet64(args.Nx,args.Ny,3,widthfakt,2,32)
if args.whichnet==65: nnet=CNet65(args.Nx,args.Ny,3,widthfakt,2,32)
if args.whichnet==66: nnet=CNet66(args.Nx,args.Ny,3,widthfakt,2,32)

if args.whichnet==100: nnet=CNet100(args.Nx,args.Ny,3,widthfakt,4,2,1.0,1,1.0,1,1)
if args.whichnet==101: nnet=CNet100(args.Nx,args.Ny,3,widthfakt,4,6,1.0,1,1.0,2,1)







if args.whichnet==11: nnet=RGNet1(args.Nx,args.Ny,3,widthfakt,4,8)
if args.whichnet==12: nnet=RGNet2(args.Nx,args.Ny,3,widthfakt,4,8)
if args.whichnet==13: nnet=RGNet3(args.Nx,args.Ny,3,widthfakt,4,8,4)


if args.whichnet==20: nnet=AANet1(args.Nx,args.Ny,widthfakt,4)
if args.whichnet==21: nnet=AANet2(args.Nx,args.Ny,widthfakt,4)
if args.whichnet==22: nnet=AANet3(args.Nx,args.Ny,widthfakt,4)

if args.whichnet==40: nnet=AANet10(args.Nx,args.Ny,widthfakt,4)

if args.whichnet==50: nnet=CANet1(args.Nx,args.Ny,widthfakt,4)



print(nnet.__class__.__module__ + "." + nnet.__class__.__name__)

arch1=expected_arch_id = CNet1.__module__ + "." + CNet1.__name__

print(arch1)


if (args.load!=None):
  nnet=RNet.load(args.load)

tdev='cpu'
  
nnet.to(tdev)



phi=torch.randn(Nb,Nx,Ny,device=tdev)

print('using ',Nb,Nx,Ny,phi.shape)


time=np.random.rand()
camp=np.sqrt(config.gint(time))
time=torch.full([Nb],camp)
time=time.to(tdev)
Relotte,grad=nnet.forward(phi,time)
step=1e-7

b=0

def nnn(aa):
   return nnet.net(aa.unsqueeze(0)).squeeze()

#ez mukodik
if 1==0:
     intensor=torch.cat([phi.view(-1),torch.tensor(camp).view(1)],dim=0)
     print(intensor.unsqueeze(0))
     R1=nnn(intensor)
     print(phi)
     intensor[8]+=step
     print(phi)
     R2=nnn(intensor)
     print((R2-R1)/step)
     print(torch.func.jacrev(nnn)(intensor))


for b in range(Nb):
     for x in range(Nx):
       for y in range(Ny):
          phi[b,x,y]+=step
     #     print(phi)
          Rutana, ngrad = nnet(phi,time)
          phi[b,x,y]-=step
#          print('shap ',Rutana.shape,grad.shape)
          print(  ((Rutana-Relotte)[b]/step - grad[b,x,y]).item())
#     print(((Rutana-Relotte)/step).item(),grad[b,x,y].item(),((Rutana-Relotte)/step - grad[b,x,y]).item())

     
#print(phidat.shape,tdat.shape)
#train(nnet, phidat , tdat , timesdat,  gint, epochs=args.Nlearn , batch_size=args.Nb, lr=1e-3)
#nnet.save('tnetstate.'+str(nfile))
#torch.save(nnet.state_dict(),"tnetstate."+str(nfile))


      
# Example usage
if __name__ == "__main__" and 0 :
    batch_size, Nx, Ny = 8, 4, 4
    model = RNet(Nx, Ny)
    x = torch.randn(batch_size, Nx, Ny)
    t= torch.rand(batch_size)
    R, grad_output = model(x,t)

    print("Input shape:", x.shape)
    print("R shape:", R.shape)
    print("Output (gradient) shape:", grad_output.shape)

      








sdfssfg3
phi=np.random.rand(4,4)

for i in range(1111):
  hmc_trajectory(phi,0.05)


aa=[]
ss=[]
pp=[]
  
for i in range(100000):
  hmc_trajectory(phi,0.05)
  action=action_2d(phi)
  sqa=phisq(phi)
  aa.append(action)
  ss.append(sqa)
  pp.append(np.average(phi))
  
h1=np.histogram(aa,bins=200,density=True)
h2=np.histogram(ss,bins=200,density=True)
h3=np.histogram(pp,bins=200,density=True)

h1a=h2[0]
h1x=h2[1]

for i in range(len(h1x)-1):
  print(h1x[i],h1a[i])
