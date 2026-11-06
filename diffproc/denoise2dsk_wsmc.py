import torch
import torch.nn as nn
import numpy as np
import random

from nns import RNet

from torchhmc import taction_2d


#    gaussfakt=0.5/gint(1.0)

#the convention here is that t goes from 1 to 0 during the denoising process
    
#this is not called yet
def give_lnP(model,phi,kappa,sklambda,t,gaussfakt):
    if (t==1.0):
#        give_gaussian
        temp=torch.sum(torch.mul(phi,phi),(1,2))
#       sum over lattice 
        return temp*(-gaussfakt)
    if (t==0.0):
        return taction_2d(phi,kappa,sklamda)

    R,grad =model.forward(phi,times)
    return R



def one_improved_step_wweights(model,phi,weights,t,tp1,gint,goft,kappa,sklambda,Njacob=1,exactdet=0):
#    phi=phi.double()
    Nb=phi.size()[0]
    ddev=next(model.parameters()).device
    tstep=t-tp1
 #   print("tstep ",tstep)
    currentamp=np.sqrt(gint(t))
    times=torch.full((Nb,),currentamp)
    times=times.to(ddev)
    currentamp2=np.sqrt(gint(tp1))
    times2=torch.full([Nb],currentamp2)
    timesp=times2.to(ddev)
    noise=torch.randn(phi.shape,device=ddev)
#      print("times ",t,times.device,Nb,Nx,Ny)
    R, grad = model.forward(phi,times)
    if (t==1.0 and True):
#        print("now replacing R")
        temp=torch.sum(torch.mul(phi,phi),(-1,-2))
        R=temp*(-0.5/gint(1.0))
    noiseampsq=gint(t)-gint(tp1)
    if (noiseampsq<0.0):
        print("baj van, nem csokken az ido")
        ghghghghg
    nfaktexact=np.sqrt(noiseampsq)
        
    g=goft(t)
    dfakt=g*g*tstep
    nfakt=g*np.sqrt(tstep)
    gp=goft(t-tstep)
    dfaktp=gp*gp*tstep
    nfaktp=gp*np.sqrt(tstep)
    link=0.0
    midphi= phi*(1.0+link*tstep)+ grad* dfakt + noise*nfakt

    Rp, gradp = model.forward(midphi,timesp)
    newphi=phi+ grad*(dfakt*0.5)+gradp*(dfaktp*0.5)+ noise*(nfaktexact)

    
    for iter in range(5):
         Rp, gradp = model.forward(newphi,timesp) 
         newphi=phi+ grad*(dfakt*0.5)+gradp*(dfaktp*0.5)+ noise* (nfaktexact )

    if (exactdet==0):  newR, newgrad = model.forward(newphi,timesp)

    if (exactdet==1):  newR, newgrad, logdetjac = model.forward_logdetjac(newphi,timesp,-0.5*dfaktp)

    if (exactdet==2) or exactdet==4:
         newR,_,jaclap = model.forward_wdiv(newphi,timesp,Njacob)
    if (exactdet==3):  newR,_,jaclap,jaclap2 = model.forward_wdiv(newphi,timesp,n_samples=Njacob,jacord=2)

    if (tp1==0.0 and True):
        newR=taction_2d(newphi,kappa,sklambda)*(-1.0)

        
    etasq=torch.sum(torch.mul(noise,noise),(-1,-2))
    backeta= phi-newphi
    backesq= torch.sum(torch.mul(backeta,backeta),(-1,-2))
    backf=nfaktexact

# we also need Jacobian!

    wfl=newR-R+etasq*0.5-backesq*(0.5/(backf*backf)) 
    if (exactdet==1): wfl-=logdetjac
    if (exactdet==2): wfl+=jaclap*(gp*gp*tstep*0.5)

    mm=torch.mean(wfl)
    wfl-=mm

    weights*=torch.exp(wfl)
    return newphi

def one_step_wweights(model,phi,weights,t,tp1,gint,goft,kappa,sklambda):
    Nb=phi.size()[0]
    ddev=next(model.parameters()).device
    tstep=t-tp1
    currentamp=np.sqrt(gint(t))
    times=torch.full((Nb,),currentamp)
    times=times.to(ddev)
    noise=torch.randn(phi.shape,device=ddev)
#      print("times ",t,times.device,Nb,Nx,Ny)
    R, grad = model.forward(phi,times)
    if (t==1.0 and True):
#        print("now replacing R")
        temp=torch.sum(torch.mul(phi,phi),(1,2))
        R=temp*(-0.5/gint(1.0))
    g=goft(t)
    dfakt=g*g*tstep
    nfakt=g*np.sqrt(tstep)
    noiseampsq=gint(t)-gint(tp1)
    if (noiseampsq<0.0):
        print("baj van, nem csokken az ido")
        ghghghghg
    nfaktexact=np.sqrt(noiseampsq)

    
    currentamp2=np.sqrt(gint(tp1))
    times2=torch.full([Nb],currentamp2)
    times2=times2.to(ddev)
    
    newphi=phi+ grad * dfakt + noise*nfakt
    newR, newgrad = model.forward(newphi,times2)
    if (tp1==0.0 and True):
        newR=taction_2d(newphi,kappa,sklambda)*(-1.0)

    etasq=torch.sum(torch.mul(noise,noise),(1,2))
    backeta= phi-newphi
    backesq= torch.sum(torch.mul(backeta,backeta),(1,2))
#    backf=(nfakt+nfaktp)*0.5
    backf=nfakt
#    backf=nfaktexact

    wfl=newR-R+etasq*0.5-backesq*(0.5/(backf*backf))
    mm=torch.mean(wfl)
    wfl-=mm
    
    weights*=torch.exp(wfl)
    return newphi

def ESS(weights):
    wsum=torch.sum(weights)
    wwsum=torch.sum(torch.mul(weights,weights))
    return wsum*wsum/wwsum

    

def denoise_using_smc_weights(model,Nb,kappa,sklambda,nstep,gint,goft,improved=0,resamplelim=0.0,Nrejuv=5,deltatl=0.025,nfile=0,finalresamp=1,Njacob=1,exactdet=0):
    initw=1.0/Nb
    ddev=next(model.parameters()).device
#    datatype=next(model.parameters()).dtype
#    print("datatype is",datatype)
    weights=torch.full((Nb,),initw,device=ddev)
    Nx=model.Nx
    Ny=model.Ny
    initamp=np.sqrt(gint(1.0))
    acc=torch.full((4,),0.0)
    phi=torch.randn(Nb,Nx,Ny,device=ddev)*initamp

    if nfile>0:
      essfile=open("ESS."+str(nfile),"w")

    
#    print("initamp ",gint(1.0))
    t=1.0
    tstep=1.0/nstep
    i=0
    
    while i<nstep-1:
        curESS=ESS(weights)
        print("elotte" , t,ESS(weights).item(),torch.mean(phi,(1,2))[0].item())
        if (nfile>0):
            print(1.0-t,ESS(weights).item(),torch.mean(phi,(1,2))[0].item(),file=essfile)
        
        if (curESS/Nb<resamplelim):
           print("resampling")
           accarr=[0,0]
           phi,weights=resample_array(phi,weights,1) 
           if Nrejuv>0: phi=langevin_update_rejuv(model,phi,Nb,t,Nrejuv,goft,gint,deltatl,accarr)
           print("langaccrej ",accarr[0],accarr[1])
           if nfile>0:
               print("#langa",accarr[0].item(),accarr[1].item(),accarr[0].item()+accarr[1].item(),file=essfile)


        
        if (improved==0): phi=one_step_wweights(model,phi,weights,t,t-tstep,gint,goft,kappa,sklambda)
        if (improved==1): phi=one_improved_step_wweights(model,phi,weights,t,t-tstep,gint,goft,kappa,sklambda,Njacob=Njacob,exactdet=exactdet)
        wsum=torch.sum(weights)
        weights=weights/wsum
        t-=tstep
        i+=1

    if (improved==0): phi=one_step_wweights(model,phi,weights,t,0.0,gint,goft,kappa,sklambda)
    if (improved==1): phi=one_improved_step_wweights(model,phi,weights,t,0.0,gint,goft,kappa,sklambda,Njacob=Njacob,exactdet=exactdet)
    
    print("final ESS ",Nb,Nx,Ny,ESS(weights).item())
#    print("initamp",initamp)
    if (nfile>0):
       print("1.0",ESS(weights).item(),torch.mean(phi,(1,2))[0].item(),file=essfile)
       essfile.flush()
       
    if finalresamp: phi,weights=resample_array(phi,weights,1)

    if finalresamp==0: return phi,acc,weights
    
    return phi,acc

def bracket_t(nogt,gint):
    lower=0.0
    upper=1.0
    while upper-lower>1e-5:
       mid=(upper+lower)*0.5
       midv=gint(mid)
       if nogt>midv: lower=mid
       else: upper=mid
    return (lower+upper)*0.5
       
def resample_array(phi,weights,whichsmp=1):
    Np=phi.size()[0]
    relist=resample_list(Np,weights,whichsmp)

    newphi=torch.full_like(phi,0.0)

    for i in range(Np):
        newphi[i]=phi[relist[i]]

    weights=torch.full_like(weights,1.0/Np)

    return newphi,weights



def resample_list(newNp,weights,whichsmp=1):
    if (whichsmp==1): return resample_list_systematic(newNp,weights)
    Np=weights.size()[0]
    wsum=0.0
    cumw=torch.full_like(weights,0.0)
    ide=[0]*Np
    
    for i in range(Np):
        wsum+=weights[i]
        cumw[i]=wsum

    for i in range(Np):
        where = random.uniform(0.0,wsum)
        k2a=0
        k2f=Np-1

        while (k2f-k2a>1):
            middle=int((k2f+k2a)*0.5)
            if (cumw[middle].item()<where):  k2a=middle
            else: k2f=middle
        k2=k2f
        if (cumw[k2-1].item()>where): k2-=1
        ide[i]=k2

    return ide

def resample_list_systematic(newNp,weights):
    Np=weights.size()[0]
    wsum=0.0
    cumw=torch.full_like(weights,0.0)
    ide=[0]*Np
    
    for i in range(Np):
        wsum+=weights[i]
        cumw[i]=wsum

    where1=random.uniform(0.0,1.0/Np)

    for i in range(Np):
        where=(where1+float(i)/Np)*wsum
        k2a=0
        k2f=Np-1

        while (k2f-k2a>1):
            middle=int((k2f+k2a)*0.5)
            if (cumw[middle]<where):  k2a=middle
            else: k2f=middle
        k2=k2f
        if (cumw[k2-1]>where): k2-=1
        ide[i]=k2
    return ide

def langevin_update_rejuv(model,phi,Nb,t,Nlang,goft,gint,deltatl,accarr):
         ddev=next(model.parameters()).device
         camp=np.sqrt(gint(t))
         times=torch.full([Nb],camp)
         times=times.to(ddev)
         g=goft(t)
         ldfakt=g*g*deltatl*0.5
         lnfakt=g*np.sqrt(deltatl)
         la=0
         lr=0                          
         for l in range(Nlang):
            noise=torch.randn(phi.shape,device=ddev)
            nsqsum=torch.sum(noise*noise,(1,2))
            Rold, grad= model.forward(phi,times)

            newphi=phi+ grad*ldfakt + noise *lnfakt

            Rnew,grad2=model.forward(newphi,times)
            backnoise=newphi+grad2*ldfakt-phi
            bsqsum=torch.sum( backnoise*backnoise , (1,2) ) *(1.0/lnfakt)**2

            chance= torch.exp( Rnew-Rold + (nsqsum - bsqsum)*0.5)
            randvec=torch.rand([Nb],device=ddev)
            mask= randvec < chance
            na=torch.sum(mask)
            la+=na
            lr+=Nb-na
            phi[mask]=newphi[mask]
         accarr[0]=la
         accarr[1]=lr
         return phi

