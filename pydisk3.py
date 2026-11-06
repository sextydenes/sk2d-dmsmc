#!/usr/bin/python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.optim.swa_utils import AveragedModel

import torchhmc
from torchhmc import thmc_trajectory_2dsk
from torchhmc import taction_2d
from torchhmc import tforce_2d


import torch.nn.functional as F

import argparse

parser=argparse.ArgumentParser(description='program to teach the neural network')

import config

parser.add_argument("-Nx",type=int,help="Nx",default=4)
parser.add_argument("-Ny",type=int,help="Ny",default=4)
parser.add_argument("-Nsamp",type=int,help="how many configs for training - this is not really used",default=2000)
parser.add_argument("-Nlearn",type=int,help="how many training steps",default=1000)
parser.add_argument("-load",type=str,help="load the state of the nn")
parser.add_argument("-actfact",type=float,help="faktor of actionfit in the loss",default=0.0)
parser.add_argument("-actfact2",type=float,help="faktor of actionfit2 in the loss",default=0.0)
parser.add_argument("-whichnet",type=int,help="what kind of nn",default=1)
parser.add_argument("-Nlayer",type=int,help="what kind of nn",default=1)
parser.add_argument("-kernelsize",type=int,help="what kind of nn",default=3)
parser.add_argument("-learningrate",type=float,help="what kind of nn",default=1e-3)
parser.add_argument("-whichopt",type=int,help="1: adamax, 2: sgd , 3:adam ",default=1e-3)
parser.add_argument("-Nfin",type=int,help="Nfin, final net width",default=1024)
parser.add_argument("-Nlap",type=int,help="Nlap, laplacesample",default=0)
parser.add_argument("-Ngradcoll",type=int,help="Ngradcoll, how many steps to collect gradient",default=1)
parser.add_argument("-ampuniform",help="training times are uniform in the amplitude, not in time",action="store_true")
parser.add_argument("-advfit",help="training changes with epoch",action="store_true")
parser.add_argument("-timefit",help="training with one config, only times change",action="store_true")
parser.add_argument("-timefitexp",help="training with one config, time distrib is exp(timefitexp*ln(uniform))",type=float,default=0.0)
parser.add_argument("-average",help="average model, useful if loss is already close to minimum. nonzero value gives the epoch when averaging is started",type=int,default=0)
parser.add_argument("-doubleprec",help="use double precision numbers",action="store_true")


args=config.add_defargs(parser)

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

torch.set_default_dtype(torch.float64)  
#torch.set_default_device(tdev)  # break dataloader

def goft(t):
  return np.power(explsigma,t)



def gint(t):
  return (np.power(explsigma,2.0*t)-np.exp(-2.0*link*t))/(2.0*(link+np.log(explsigma)))

def gint_nolinforce(t):
  return 0.5/np.log(explsigma)*(np.power(explsigma,2.0*t)-1.0)



def gsqt(t):
    return torch.pow(explsigma,2.0*t)


def tgint(t):
  return (torch.pow(explsigma,2.0*t)-torch.exp(-2.0*link*t))*gintfakt

def tgint_deriv(t):
  return (torch.pow(explsigma,2.0*t)*2.0*np.log(explsigma) +2.0 *link *torch.exp(-2.0*link*t) )*gintfakt

def sqrttgint(t):
  return torch.sqrt((torch.pow(explsigma,2.0*t)-torch.exp(-2.0*link*t))*gintfakt)

def sqrttgint_tderiv(t):
  omega=torch.sqrt((torch.pow(explsigma,2.0*t)-torch.exp(-2.0*link*t))*gintfakt)
#  fakt1=(torch.pow(explsigma,2.0*t)*(2.0*np.log(explsigma))+torch.exp(-2.0*link*t)*2.0*link)*gintfakt
  gsqr=torch.pow(explsigma,2.0*t)

#  return fakt1*(0.5/explsigma)
  return gsqr*0.5/omega



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


def create_traindata(Nbig,Nx,Ny,mydev):
  torch_phidat = torch.randn(Nbig, Nx, Ny,device=mydev)
  torch_noiseamp = torch.empty(Nbig)
  torch_times = torch.empty(Nbig)
  for i in range(1111):
    thmc_trajectory_2dsk(torch_phidat,0.02,kappa,sklambda)
  for i in range(Nbig):
    time=np.random.rand()
    torch_noiseamp[i]=np.sqrt(gint(time))
    torch_times[i]=time
  torch_noiseamp=torch_noiseamp.to(mydev)
  torch_times=torch_times.to(mydev)
#  print("de sikerult",torch_noiseamp.device,tdev)
  
  return torch_phidat, torch_noiseamp , torch_times

def create_phidat(Nbig,Nx,Ny,thermt=1111):
    torch_phidat = torch.randn(Nbig, Nx, Ny,device=tdev)
    for i in range(thermt):
       thmc_trajectory_2dsk(torch_phidat,0.02,kappa,sklambda)
    return torch_phidat

def evolve_phidat(tp,Ntraj=20):
   for i in range(Ntraj):
     thmc_trajectory_2dsk(tp,0.02,kappa,sklambda)
   return tp

def evolve_phidat_gauss(tp,Ntraj=20):
   Nb,Nx,Ny= tp.shape
   tp = torch.randn( Nb, Nx,Ny, device=tdev)
   return tp


from nns import *

tmax=1.01


from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

def train(model,  Nx, Ny , epochs=10, batch_size=32, lr=1e-3):
    """
    model: instance of RNet
    x_fields: Tensor (Nlarge, Nx, Ny) - input 2D fields
    t_scalars: Tensor (Nlarge,) - input scalar times
    loss_fn: function(gradients, x_fields, t_scalars) -> scalar loss
    epochs: number of training epochs
    batch_size: minibatch size
    lr: learning rate
    """
#    torch.set_default_device('cpu')
#    dataset = TensorDataset(x_fields, t_scalars, times_scalars)

#    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#    torch.set_default_device(tdev)
    optimizer = optim.Adamax(model.parameters(), lr=lr)
#    if (args.whichopt==1):  optimizer = optim.Adamax(model.parameters(), lr=lr)
    if (args.whichopt==2):  optimizer = optim.SGD(model.parameters(), lr=lr)
    if (args.whichopt==3):  optimizer = optim.Adam(model.parameters(), lr=lr)


    if (args.timefit): dataset=create_phidat(1,Nx,Ny)
    else:  dataset=create_phidat(batch_size,Nx,Ny)


    
    Nl=200
    lossbin=tmax/Nl
    hits=np.zeros(Nl)
    losses=np.zeros(Nl)
    
    zerotimes=torch.full([batch_size],config.gint(0.0))
    t_null=zerotimes.to(tdev)
    actionlossfakt=args.actfact
    difffakt=1.0-args.actfact-args.actfact2

    finaltimes=torch.full([batch_size],config.gint(1.0))
    t_fin=finaltimes.to(tdev)

    learnfile=open('learnfile.' + str(args.nfile), 'w')
    sjfile=open('sjoe.'+str(args.nfile),'w')
    sj2file=open('sjoe2.'+str(args.nfile),'w')

    acalc=actioncalc_2d()

    losshistn=300
    losshist=torch.zeros(losshistn)
    losshistavrn=torch.zeros(losshistn)
    collect_losshist=0   # this is already implemented once: diffa

    hbins=200
    bins=torch.linspace(-4,4,hbins)
    sum_counts = torch.zeros(len(bins) - 1)
    bin_width = bins[1] - bins[0]
    istep=0
        
    for epoch in range(epochs):
        total_loss = 0.0

        if (args.average and epoch>args.average):
          model_avg.update_parameters(nnet)
          if epoch %50 ==0:
            nnet.load_state_dict(model_avg.module.state_dict())
            nnet.save('atnetstate.'+str(int(epoch))+"."+str(nfile))


        if args.average==0 and epoch % 5 ==0:
          nnet.save('tnetstate.'+str(int(epoch))+"."+str(nfile))

        if (args.advfit): model.fitepoch(epoch)
          

        for xyz in range(1):
            dataset=evolve_phidat(dataset)

            

            
            phivals=dataset.mean((1,2)).to('cpu')

            counts, _ = torch.histogram(phivals, bins=bins)
            sum_counts += counts

            x_batch = dataset.detach().clone()

            if (args.timefit):
                 x_batch=x_batch.expand(batch_size, -1, -1)
             
            x_batch = x_batch.requires_grad_(True)  # Needed for jacrev gradients
            Nbatch=x_batch.shape[0]
            if istep==0: 
                optimizer.zero_grad()

            noise = torch.randn(x_batch.shape,device=tdev)
            Nb=x_batch.shape[0]



            
            if args.ampuniform:
                sigt_b2=torch.rand(Nb,device=tdev)*(np.sqrt(gint(1.0))*1.01)
                dampfakt=torch.full_like(sigt_b2,1.0).view(Nbatch,1,1)
                if (link>0.0):
                   print("this is unimplemented")
            else:
              times_b2=torch.rand(Nb,device=tdev)*tmax
#            times_b2=0.01*torch.rand(Nb,device=tdev)*tmax
              dampfakt=torch.exp(-link*times_b2).view(Nbatch,1,1)
              sigt_b2=sqrttgint(times_b2)

              #assuming tmax=1 and no damping
            if args.timefitexp!=0.0:
              uniform=torch.exp(torch.log(torch.rand(Nb,device=tdev))*args.timefitexp)*1.01
#              print(uniform)
              sigt_b2=sqrttgint(uniform)


              

            noised = x_batch*dampfakt-noise*sigt_b2.view(Nbatch,1,1)            
            R, grad = model(noised, sigt_b2)
            
            loss=F.mse_loss(noise, grad*sigt_b2.view(Nbatch,1,1), reduction='mean')

            nullR, grad2 = model(x_batch,t_null)
            actions=taction_2d(x_batch,kappa,sklambda)
            act=nullR+actions            
            loss2= (act*act).mean()

            
            if collect_losshist:
               ttt = noise-grad*sigt_b2.view(Nbatch,1,1)
               losstt= ttt*ttt
               indices=(times_b2*losshistn).to(torch/int32)
               losshistavrn[indices]+=1
               losshist[indices]+= losstt[indices]

            # no -1.0 here R = - act
            actforce=tforce_2d(x_batch,1.0,kappa,sklambda)

            loss3=F.mse_loss(actforce,grad2)
            tdstep=0.000001
            torch.set_printoptions(precision=14)
            tsteps=torch.tensor(0.0001)
            
            if 0: # epoch>20:
                if args.Nlap==0: lap=model.wasteful_laplacian(x_batch,sigt_b2)
                elif args.Nlap<0: lap=model.laplacian_discreet(x_batch,sigt_b2,tsteps)
                else: lap=model.laplacian_noisy(x_batch,sigt_b2,args.Nlap)

                gradsq = model.forward_gradsq(x_batch,sigt_b2)
                tder=model.t_derivative(x_batch,sigt_b2)*sqrttgint_tderiv(times_b2)
#                tder3=model.t_derivative3(x_batch,sigt_b2)*sqrttgint_tderiv(times_b2)
#                tder2=model.t_derivative2(x_batch,sigt_b2)*gsqt(times_b2)
#                tder22=model.t_derivative22(x_batch,sigt_b2)*gsqt(times_b2)
#                print("tder tder2 " ,tder-tder2,tder2,tder22,tder3)

#                Re,grade=model.forward(x_batch,sigt_b2)
#                times_b2+= tdstep
#                sigt_b2=sqrttgint(times_b2)
#                Rv,gradv=model.forward(x_batch,sigt_b2)
#                deriv=(Rv-Re)/tdstep
#                print("tdertest ",tder,deriv) # ez jo

                if 0: # works if stepsize not too small
                    torch.set_printoptions(precision=14)
                    lap2=model.laplacian_discreet(x_batch,sigt_b2,0.0001)
                    lap4=model.lap_disc_single(x_batch[0],sigt_b2[0],0.0001)
                    lap3=model.laplacian_noisy_single2(x_batch[0],sigt_b2[0])
                    lap5=model.wasteful_laplacian(x_batch[0],sigt_b2[0])
                    print("tder lap " , lap , lap2, lap3 ,lap4, lap5)
                
                if 0:  # this works
                    ggtt1=model.grad_disc_single(x_batch[0],sigt_b2[0],0.0000001)
                    ggtt2=model.gradtest(x_batch[0],sigt_b2[0])
                    joe=((ggtt1-ggtt2)*(ggtt1-ggtt2)).sum()
                    print("joe grad ",joe)



                lap+=gradsq
    
                lap*=gsqt(times_b2)*0.5
                

                sigmasq=(sigt_b2*sigt_b2+1.0)

                tder-=Nx*Ny*0.5*torch.pow(sigmasq,-1)*gsqt(times_b2)
                
                lap-=tder 



              #print(lap)
                loss4=(lap*lap).sum()
            else:
              loss4=torch.tensor(0.0)
                
#            finR, grad3 = model(x_batch,t_null)
#            noise2 = torch.randn(x_batch.shape,device=tdev)
#            fin_x= x_batch+noise2*np.sqrt(gint(1.0))
#            F.mse_loss(fin_x/initsigma, grad2, reduction='mean')

            
            actions=taction_2d(x_batch,kappa,sklambda)
            act=nullR+actions

            bigloss=difffakt*loss+actionlossfakt*loss3 + args.actfact2 * loss2
        
            
            bigloss.backward()
            istep+=1
            if istep==args.Ngradcoll:
               optimizer.step()
               istep=0
            
            
            if epoch>epochs*0.85:
                 for b in range(Nb):
                    print(actions[b].item(),nullR[b].item(),file=sjfile)
                 sjfile.flush()
                 print(actions[0].item(),nullR[0].item(),acalc(x_batch[0],kappa,sklambda).item(),file=sj2file)
            
            if False and epoch>epochs*0.89:
              for i in range(times_batch.shape[0]):
                 time=times_b2[i].item()
                 kisloss=F.mse_loss(noise[i],grad[i]*sigt_b2[i],reduction='mean')
                 mycol=int(time/lossbin)
                 hits[mycol]+=1
                 losses[mycol]+=kisloss.item()
            

            total_loss += loss.item() * x_batch.size(0)  # sum over batch

        avg_loss = bigloss  # / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}",loss.item(),loss2.item(),loss3.item(),loss4.item(),actionlossfakt)
        print(epoch,bigloss.item(),loss.item(),loss2.item(),loss3.item(),loss4.item(),file=learnfile)
        learnfile.flush()


    with open('diffa.'+str(args.nfile), 'w') as f:
      for i in range(Nl):
        if hits[i]>0: print(i,i*lossbin,losses[i]/hits[i],file=f)

    pdf = sum_counts / (torch.sum(sum_counts) * bin_width) 
    hh=open('hmchist.'+str(args.nfile),'w') 
    for i in range(hbins-1):
        print(bins[i].item(),pdf[i].item(),file=hh)
        
if (args.Nsamp % args.Nb):  args.Nsamp=args.Nb*int(args.Nsamp/args.Nb)
        
#create data
#phidat,tdat,timesdat = create_traindata(args.Nsamp ,args.Nx,args.Ny)

#meansq=torch.mean(phidat*phidat).item()
#initsigma=gint(1.0)+meansq
#print("mean sq",meansq,initsigma)

if (args.doubleprec):
    torch.set_default_dtype(torch.float64)



if args.load==None:
   if args.whichnet==1: nnet=RNet(args.Nx,args.Ny,args.Nx*args.Ny*widthfakt)
   if args.whichnet==2: nnet=CNet1(args.Nx,args.Ny,args.kernelsize,widthfakt,args.Nlayer)
   if args.whichnet==3: nnet=CNet2(args.Nx,args.Ny,args.kernelsize,widthfakt,args.Nlayer,args.Nfin)
   if args.whichnet==4: nnet=CNet3(args.Nx,args.Ny,args.kernelsize,widthfakt,4*args.widthfakt,args.Nfin,kappa=kappa,sklambda=sklambda)
   if args.whichnet==5: nnet=CNet4(args.Nx,args.Ny,args.kernelsize,widthfakt,4*args.widthfakt,args.Nfin,kappa=kappa,sklambda=sklambda)
   if args.whichnet==6: nnet=CNet5(args.Nx,args.Ny,args.kernelsize,widthfakt,4*args.widthfakt,args.Nfin,kappa=kappa,sklambda=sklambda)
   if args.whichnet==7: nnet=CNet6(args.Nx,args.Ny,args.kernelsize,widthfakt,4*args.widthfakt,args.Nfin,kappa=kappa,sklambda=sklambda)
   if args.whichnet==8: nnet=CNet7(args.Nx,args.Ny,args.kernelsize,widthfakt,4*args.widthfakt,args.Nfin,kappa=kappa,sklambda=sklambda)
   if args.whichnet==9: nnet=CNet8(args.Nx,args.Ny,args.kernelsize,widthfakt,4*args.widthfakt,args.Nfin,kappa=kappa,sklambda=sklambda)

   if args.whichnet==60: nnet=CNet60(args.Nx,args.Ny,args.kernelsize,widthfakt,4*args.widthfakt,args.Nfin,gint(1.0),kappa=kappa,sklambda=sklambda)

   if args.whichnet==61: nnet=CNet61(args.Nx,args.Ny,args.kernelsize,widthfakt,4*args.widthfakt,args.Nfin,kappa=kappa,sklambda=sklambda)
   if args.whichnet==62: nnet=CNet62(args.Nx,args.Ny,args.kernelsize,widthfakt,args.Nlayer,args.Nfin)
   if args.whichnet==63: nnet=CNet63(args.Nx,args.Ny,args.kernelsize,widthfakt,4*args.widthfakt,args.Nfin,kappa=kappa,sklambda=sklambda)
   if args.whichnet==64: nnet=CNet64(args.Nx,args.Ny,args.kernelsize,widthfakt,4*args.widthfakt,args.Nfin,kappa=kappa,sklambda=sklambda)
   if args.whichnet==65: nnet=CNet65(args.Nx,args.Ny,args.kernelsize,widthfakt,args.Nlayer,args.Nfin,kappa=kappa,sklambda=sklambda)
   if args.whichnet==66: nnet=CNet66(args.Nx,args.Ny,args.kernelsize,widthfakt,args.Nlayer,args.Nfin)


   if args.whichnet==100: nnet=CNet100(args.Nx,args.Ny,args.kernelsize,widthfakt,args.Nlayer,2,1.0,1,1.0,1,1)
   if args.whichnet==101: nnet=CNet100(args.Nx,args.Ny,args.kernelsize,widthfakt,args.Nlayer,6,1.0,1,1.0,2,1)
   if args.whichnet==102: nnet=CNet100(args.Nx,args.Ny,args.kernelsize,widthfakt,args.Nlayer,32,3.0,1,1.0,2,1)
   if args.whichnet==103: nnet=CNet100(args.Nx,args.Ny,args.kernelsize,widthfakt,args.Nlayer,16,3.0,1,1.0,3,1)
   


   
   if args.whichnet==11: nnet=RGNet1(args.Nx,args.Ny,3,widthfakt,4,4)
   if args.whichnet==12: nnet=RGNet2(args.Nx,args.Ny,3,widthfakt,4,4)


   if args.whichnet==20: nnet=AANet1(args.Nx,args.Ny,widthfakt,4)
   if args.whichnet==21: nnet=AANet2(args.Nx,args.Ny,widthfakt,4)
   if args.whichnet==22: nnet=AANet3(args.Nx,args.Ny,widthfakt,4)
   if args.whichnet==23: nnet=AANet4(args.Nx,args.Ny,widthfakt,4)
   if args.whichnet==24: nnet=AANet5(args.Nx,args.Ny,widthfakt,4)
   if args.whichnet==25: nnet=AANet6(args.Nx,args.Ny,widthfakt,4)

   if args.whichnet==30: nnet=GaussNet(args.Nx,args.Ny,tgint,1.0)
 

   if args.whichnet==40: nnet=AANet10(args.Nx,args.Ny,widthfakt,4)
   if args.whichnet==41: nnet=AANet11(args.Nx,args.Ny,widthfakt,4)

   if args.whichnet==50: nnet=CANet1(args.Nx,args.Ny,widthfakt,4)

   
#torch.nn.init.normal_(nnet,mean=0.0,std=0.000001)

   
if (args.load!=None):
  checkpoint = torch.load(args.load)
  arch20=expected_arch_id = AANet1.__module__ + "." + AANet1.__name__
  arch21=expected_arch_id = AANet2.__module__ + "." + AANet2.__name__
  arch22=expected_arch_id = AANet3.__module__ + "." + AANet3.__name__
  arch23=expected_arch_id = AANet4.__module__ + "." + AANet4.__name__
  arch24=expected_arch_id = AANet5.__module__ + "." + AANet5.__name__
  arch25=expected_arch_id = AANet6.__module__ + "." + AANet6.__name__

  arch40=expected_arch_id = AANet10.__module__ + "." + AANet10.__name__
  arch41=expected_arch_id = AANet11.__module__ + "." + AANet11.__name__

  arch50=expected_arch_id = CANet1.__module__ + "." + CANet1.__name__
  
  arch11=expected_arch_id = RGNet1.__module__ + "." + RGNet1.__name__
  arch12=expected_arch_id = RGNet2.__module__ + "." + RGNet2.__name__
  arch60=expected_arch_id = CNet60.__module__ + "." + CNet60.__name__
  arch61=expected_arch_id = CNet61.__module__ + "." + CNet61.__name__
  arch62=expected_arch_id = CNet62.__module__ + "." + CNet62.__name__
  arch63=expected_arch_id = CNet63.__module__ + "." + CNet63.__name__
  arch64=expected_arch_id = CNet64.__module__ + "." + CNet64.__name__
  arch65=expected_arch_id = CNet65.__module__ + "." + CNet65.__name__
  arch66=expected_arch_id = CNet66.__module__ + "." + CNet66.__name__
  arch9=expected_arch_id = CNet8.__module__ + "." + CNet8.__name__
  arch8=expected_arch_id = CNet7.__module__ + "." + CNet7.__name__
  arch7=expected_arch_id = CNet6.__module__ + "." + CNet6.__name__
  arch6=expected_arch_id = CNet5.__module__ + "." + CNet5.__name__
  arch5=expected_arch_id = CNet4.__module__ + "." + CNet4.__name__
  arch4=expected_arch_id = CNet3.__module__ + "." + CNet3.__name__
  arch3=expected_arch_id = CNet2.__module__ + "." + CNet2.__name__
  arch2=expected_arch_id = CNet1.__module__ + "." + CNet1.__name__
  arch1=expected_arch_id = RNet.__module__ + "." + RNet.__name__
  if arch1== checkpoint['arch_id']: 
       nnet=RNet.load(args.load)
  elif arch11== checkpoint['arch_id']:
       nnet=RGNet1.load(args.load)
  elif arch12== checkpoint['arch_id']:
       if (args.whichnet==13):
           nnet=RGNet3.partload(args.load)
       else:  nnet=RGNet2.load(args.load) 
  elif arch2== checkpoint['arch_id']:
       nnet=CNet1.load(args.load)
  elif arch3== checkpoint['arch_id']:
       nnet=CNet2.load(args.load)
  elif arch4== checkpoint['arch_id']:
       nnet=CNet3.load(args.load)
  elif arch5== checkpoint['arch_id']:
       nnet=CNet4.load(args.load)
  elif arch6== checkpoint['arch_id']:
       nnet=CNet5.load(args.load)
  elif arch7== checkpoint['arch_id']:
       nnet=CNet6.load(args.load)
  elif arch8== checkpoint['arch_id']:
       nnet=CNet7.load(args.load)
  elif arch9== checkpoint['arch_id']:
       nnet=CNet8.load(args.load)
  elif arch60== checkpoint['arch_id']:
       nnet=CNet60.load(args.load)
  elif arch61== checkpoint['arch_id']:
       nnet=CNet61.load(args.load)
  elif arch62== checkpoint['arch_id']:
       nnet=CNet62.load(args.load)
  elif arch63== checkpoint['arch_id']:
       nnet=CNet63.load(args.load)
  elif arch64== checkpoint['arch_id']:
       nnet=CNet64.load(args.load)
  elif arch65== checkpoint['arch_id']:
       nnet=CNet65.load(args.load)
  elif arch66== checkpoint['arch_id']:
       nnet=CNet66.load(args.load)                                                
       
  elif arch20== checkpoint['arch_id']:
       nnet=AANet1.load(args.load)
  elif arch21== checkpoint['arch_id']:
       nnet=AANet2.load(args.load)
  elif arch22== checkpoint['arch_id']:
       nnet=AANet3.load(args.load)
  elif arch23== checkpoint['arch_id']:
       nnet=AANet4.load(args.load)
  elif arch24== checkpoint['arch_id']:
       nnet=AANet5.load(args.load)
  elif arch25== checkpoint['arch_id']:
       nnet=AANet6.load(args.load)
  elif arch40== checkpoint['arch_id']:
       nnet=AANet10.load(args.load)
  elif arch41== checkpoint['arch_id']:
       nnet=AANet11.load(args.load)
  elif arch50== checkpoint['arch_id']:
       nnet=CANet1.load(args.load)                                            
  else:
    print('don\'t know this net')
    xxxxxxx
  print("loaded " + nnet.__module__)


  

nnet.to(tdev)
nnet.train()

if (args.average):
  model_avg = AveragedModel(nnet)


print( args.Nb, args.Nx, args.Ny) 
train(nnet, args.Nx , args.Ny , epochs=args.Nlearn , batch_size=args.Nb, lr=args.learningrate)


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

      







