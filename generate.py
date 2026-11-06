import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import sys
import textwrap

import torchhmc
from torchhmc import thmc_trajectory_2dsk
from torchhmc import taction_2d
import torch.nn.functional as F



from nns import *

import argparse

class CustomHelpFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith('RAW:'):
            return text[4:].splitlines()
        return super()._split_lines(text, width)


parser=argparse.ArgumentParser(description='this one should create configs. you have to load a neural net with -load ffff',formatter_class=CustomHelpFormatter)
tdev='cuda' if torch.cuda.is_available() else 'cpu'




from diffproc import denoise_using_smc_weights


import config


parser.add_argument("-Nx",type=int,help="lattice size")
parser.add_argument("-Ny",type=int,help="lattice size")
parser.add_argument("-Nconf",type=int,help="how many to generate")
parser.add_argument("-load",type=str,help="netstate filename")
parser.add_argument("-Nlang",type=int,help="how many langevin steps ",default=0)
parser.add_argument("-deltatl",type=float,help="langevin stepsizes ",default=0.03)
parser.add_argument("-gendeltat",type=float,help="generate deltatl",default=0.01)
parser.add_argument("-alla",type=int,help="1: always accept",default=0)
parser.add_argument("-minlangtime",type=float,help="langevin corrections mintime",default=0.0)
parser.add_argument("-langcorrtime",type=float,help="langevin corrections starttime for update 3",default=0.0)
parser.add_argument("-langtime",type=float,help="for langevin test",default=0.0)
parser.add_argument("-whichupd",type=int,help=textwrap.dedent('''RAW:which update in the denoising
1:
2:
3:
10: DM-SMC naive update
11: DM-SMC improved update, Njacob vectors
12: DM-SMC improved update, exact jacobian
20: DM-SMC naive update, no final resampling
21: DM-SMC improved update, Njacob vectors, no final resampling'''),default=1)
parser.add_argument("-initsig",type=float,help="sigma sq average of the distribution ",default=0.0)
parser.add_argument("-hack",type=float,help="hack initsigfakt ",default=0.0)

parser.add_argument("-print1dhist",help="print measure for 1d distribution",action="store_true")
parser.add_argument("-doubleprec",help="use double precision numbers",action="store_true")
parser.add_argument("-floatprec",help="use single precision numbers",action="store_true")

parser.add_argument("-smcsteps",type=int,help="how many steps to do smc",default=20)
parser.add_argument("-resamplimit",type=float,help="ESS decrease to resample",default=0.0)
parser.add_argument("-Njacob",type=int,help="How many vectors to use to estimate jacobian for improved updates",default=1)



config.add_defargs(parser)

args=parser.parse_args()

if args.dev!=None:
  tdev=args.dev

    
Nconf=args.Nconf
if Nconf==None: Nconf=100


sklambda=config.sklambda
kappa=config.kappa
explsigma=config.explsigma
link=config.link
nfile=config.nfile

checkpoint = torch.load(args.load,map_location=tdev)



arch2=expected_arch_id = CNet1.__module__ + "." + CNet1.__name__
arch1=expected_arch_id = RNet.__module__ + "." + RNet.__name__

if arch2== checkpoint['arch_id']:
       nnet=CNet1.load(args.load)
elif arch3== checkpoint['arch_id']:
       nnet=CNet2.load(args.load)

       
else:
  print('don\'t know this net')
  xxxxxxx

if (1==0):
    for layer in nnet.modules():
        if hasattr(nnet,'reset_parameters'):
            layer.reset_parameters()

  


            
if args.Nx is not None:
  nnet.Nx=args.Nx
if args.Ny is not None:
  nnet.Ny=args.Ny
nnet.eval()


print("config:",nfile,link,explsigma,kappa,sklambda,"Nxy ",nnet.Nx,nnet.Ny)
#print("c2: ",config.nfile,args.nfile)




print("model loaded")

if (args.doubleprec):
    nnet.double()
    torch.set_default_dtype(torch.float64)

if (args.floatprec):
    nnet.float()
    torch.set_default_dtype(torch.float32)

    
nnet.to(tdev)
#datatype=torch.float64
#nnet.to(datatype)

#print(next(nnet.parameters()).device)
phi=torch.randn(args.Nb,nnet.Nx,nnet.Ny,device=tdev) #,dtype=datatype)
 
oneovervol=1.0/(nnet.Nx*nnet.Ny)
#with torch.no_grad(): 
#   phi=denoise_nconfs_2dsk(nnet,args.Nb,config.kappa,config.sklambda,args.gendeltat,config.gint,config.goft)
#   phi,accreja=denoise_nconfs_2dsk_lang_accrej(phi,nnet,args.Nb,config.kappa,config.sklambda,args.gendeltat,config.gint,config.goft,args.Nlang,args.deltatl,1,args.minlangtime,link=link)
   
aa=[]
ss=[]
pp=[]

if args.print1dhist:
   rfile=open("Raction."+str(nfile),"w")

   xx=-30.0
   Z=0.0
   RZ1=0.0
   while (xx<30.0):
       phix=torch.full([1,1,1],xx)
       xx+=0.01
       Z+=np.exp(-taction_2d(phix,config.kappa,config.sklambda)[0].item())
       times=torch.full([1],np.sqrt(config.gint(0.0)))
       R,grad=nnet.forward(phix.to(tdev),times.to(tdev))
       RZ1+=np.exp(R[0].item())
   fakt=100.0/Z
   RZ1fakt=100.0/RZ1

   
   x=-35.0
   while x<35.0:
     print(x," ",end="",file=rfile)
     phi=torch.full([1,1,1],x)
     for t in [0.0,0.1,0.5,0.9,1.0]:
       phi[0]=x
       times=torch.full([1],np.sqrt(config.gint(t)))
       R,grad=nnet.forward(phi.to(tdev),times.to(tdev))

       print(np.exp(R[0].item())*RZ1fakt," ",end="",file=rfile)


   
       
     print(np.exp(-taction_2d(phi,config.kappa,config.sklambda)[0].item())*fakt,file=rfile)  
   
     x+=0.1


   sys.exit(0)



arfile=open("arej."+str(nfile),"w")
histfile=open("hist."+str(nfile),"w")
genfile=open("data."+str(nfile),"w")

time=0.0
camp=np.sqrt(config.gint(time))
time=torch.full([args.Nb],camp)
time=time.to(tdev)

#nnet.geninit()

weights=[]

for i in range(Nconf):
   with torch.no_grad():   
       if (args.whichupd==10):  newphi,accreja=denoise_using_smc_weights(nnet,args.Nb,config.kappa,config.sklambda,args.smcsteps,config.gint,config.goft,resamplelim=args.resamplimit,nfile=args.nfile)
       if (args.whichupd==20):  newphi,accreja=denoise_using_smc_weights(nnet,args.Nb,config.kappa,config.sklambda,args.smcsteps,config.gint,config.goft,resamplelim=args.resamplimit,nfile=args.nfile,finalresamp=0)

       
       if (args.whichupd==11):  newphi,accreja=denoise_using_smc_weights(nnet,args.Nb,config.kappa,config.sklambda,args.smcsteps,config.gint,config.goft,improved=1,resamplelim=args.resamplimit,nfile=args.nfile,Njacob=args.Njacob,exactdet=1)
       if (args.whichupd==21):  newphi,accreja=denoise_using_smc_weights(nnet,args.Nb,config.kappa,config.sklambda,args.smcsteps,config.gint,config.goft,improved=1,resamplelim=args.resamplimit,nfile=args.nfile,finalresamp=0)


       if (args.whichupd==12):  newphi,accreja=denoise_using_smc_weights(nnet,args.Nb,config.kappa,config.sklambda,args.smcsteps,config.gint,config.goft,improved=1,resamplelim=args.resamplimit,nfile=args.nfile,Njacob=args.Njacob,exactdet=2)

       
       print(accreja[0].item(),accreja[1].item(),accreja[2].item(),accreja[3].item(),file=arfile)
       arfile.flush()
       phi=newphi
       action=taction_2d(phi,kappa,sklambda)
       phia=torch.sum(phi,(1,2))
       sqa=torch.sum(phi*phi,(1,2))
       R, grad=nnet.forward(phi,time)
#       print(action)
       for b in range(phi.shape[0]):
          aa.append(action[b].item())
          ss.append(sqa[b].item()*oneovervol)
          pp.append(phia[b].item()*oneovervol)
          if (len(weights)==0):
               print(action[b].item(),phia[b].item()*oneovervol,sqa[b].item()*oneovervol,R[b].item(),file=genfile)
          else:
               print(action[b].item(),phia[b].item()*oneovervol,sqa[b].item()*oneovervol,R[b].item(),weights[b].item(),file=genfile)
       genfile.flush()
          

   
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
  print(h1x[i],h1a[i],h2x[i],h2a[i],h3x[i],h3a[i],file=histfile)
