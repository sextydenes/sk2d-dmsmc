
import torch


#assumes 3 indices: batch , x, y

def taction_2d(phi,kappa,sklambda):
    sq=torch.sum(torch.mul(phi,phi),(1,2))
    quad=torch.sum(torch.pow(phi,4),(1,2))
    shift = torch.roll(phi, shifts=1, dims=1)  # shift left
    shift += torch.roll(phi, shifts=1, dims=2)  # shift up

# Sum of squared differences (periodic)
    ssd = torch.sum( phi*shift , (1,2) )
    return sq*(1.0-2.0*sklambda)+quad*sklambda-ssd*2.0*kappa


def tforce_2d_add(phi,force,fakt,kappa,sklambda):
    dinpard=-2.0*(1.0-2.0*sklambda)*fakt
    dinparl=-4.0*sklambda*fakt
    dinparn=2.0*kappa*fakt

    cub=torch.pow(phi,3)
    shift = torch.roll(phi, shifts=1, dims=1)  # shift left
    shift += torch.roll(phi, shifts=-1, dims=1)  # shift left
    shift += torch.roll(phi, shifts=1, dims=2)  # shift left
    shift += torch.roll(phi, shifts=-1, dims=2)  # shift left

    force+=phi*dinpard+cub*dinparl+shift*dinparn

def tforce_2d(phi,fakt,kappa,sklambda):
    dinpard=-2.0*(1.0-2.0*sklambda)*fakt
    dinparl=-4.0*sklambda*fakt
    dinparn=2.0*kappa*fakt

    cub=torch.pow(phi,3)
    shift = torch.roll(phi, shifts=1, dims=1)  # shift left
    shift += torch.roll(phi, shifts=-1, dims=1)  # shift left
    shift += torch.roll(phi, shifts=1, dims=2)  # shift left
    shift += torch.roll(phi, shifts=-1, dims=2)  # shift left

    return phi*dinpard+cub*dinparl+shift*dinparn    
    



    
def thmc_trajectory_2dsk(phi,deltat,kappa,sklambda):
    if (len(phi.shape)!=3):
        print("3 index tensors are needed\n")
        rrrrrrr
    mom=torch.randn(phi.shape,device=phi.device)
    phic=phi.clone()
    elotte=taction_2d(phi,kappa,sklambda)+torch.sum(mom*mom,(1,2))*0.5
    nstep=int(1.0/deltat)
    for i in range(nstep):
        tforce_2d_add(phi,mom,deltat*0.5,kappa,sklambda)
        phi+=mom*deltat
        tforce_2d_add(phi,mom,deltat*0.5,kappa,sklambda)
#        aaa=(taction_2d(phi,kappa,sklambda)+torch.sum(mom*mom,(1,2))*0.5)
#        print(i,aaa-elotte)
    last_action=taction_2d(phi,kappa,sklambda)+torch.sum(mom*mom,(1,2))*0.5
    chance=torch.exp(elotte-last_action)
    randomvec=torch.rand(chance.shape,device=phi.device)
    mask= randomvec > chance 
    phi[mask]=phic[mask]
    nr=torch.sum(mask)
    return nr
    
