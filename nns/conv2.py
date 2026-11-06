import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.func import vmap, jacrev

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mact(nn.Module):
    def forward(self,x):
        return x * torch.sigmoid(x)
    

# works for 2d tensors
class actioncalc_2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,phi,kappa,sklambda):
        sqf=torch.mul(phi,phi)
        sq=torch.sum(sqf)
        quad=torch.sum(sqf*sqf)
        shift = torch.roll( phi,shifts=1,dims=0)
        shift += torch.roll( phi,shifts=1,dims=1)
        ssd= torch.sum(phi*shift)
        return sq*(1.0-2.0*sklambda)+quad*sklambda-ssd*(kappa*2.0)

    
class CNet2(nn.Module):
    def __init__(self, Nx, Ny, kernel_size, Nc , Nlayer ,Nfin):
        super().__init__()
        self.config = {
            'Nx' : Nx ,
            'Ny' : Ny ,
            'kernel_size' : kernel_size,
            'Nc' : Nc ,
            'Nlayer' : Nlayer,
            'Nfin' : Nfin
        }
        self.Nx = Nx
        self.Ny = Ny
        self.Nfin = Nfin
        self.kernel_size=kernel_size
        self.Nc = Nc
        self.Nlayer = Nlayer
        self.input_dim=Nx*Ny+1
        self.padding= (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=Nc, kernel_size=kernel_size, padding=0)

        self.inconv=[]
        for i in range(Nlayer):
            self.inconv.append(nn.Conv2d(in_channels=Nc, out_channels=Nc, kernel_size=kernel_size, padding=0))
        self.inlayers=nn.ModuleList(self.inconv)
  
        self.mixlayer=[]
        self.activ=nn.SiLU()
        for i in range(Nlayer):    
           self.mixlayer.append(nn.Conv2d(in_channels=Nc, out_channels=Nc, kernel_size=1, padding=0))
        self.mixlayers=nn.ModuleList(self.mixlayer)

        self.lastmix= nn.Conv2d(in_channels=Nc, out_channels=1, kernel_size=1, padding=0)  
#        self.pool = nn.AvgPool2d(kernel_size=(Nx,Ny),stride=(Nx,Ny),divisor_override=1)
        self.finalcalcn=1+Nc*Nlayer+1+1
        self.finlin1=nn.Linear(self.finalcalcn,Nfin)
 #       self.finlin2=nn.Linear(Nfin,Nfin)
 #       self.finlin3=nn.Linear(Nfin,Nfin)
        self.finlin4=nn.Linear(Nfin,1)
        print("finalcalcn",self.finalcalcn)

        
    def forward(self,x_batch,t_batch):
        B, Nx, Ny = x_batch.shape
        padding=self.padding
        
        def net_fn(x_field,t_scalar):
            res1=torch.sum(x_field).view(1)
            bt=t_scalar.expand([Nx,Ny])
            c2t=torch.cat([x_field.view(1,Nx,Ny), bt.view(1,Nx,Ny)], dim=0)
            
            p1=F.pad(c2t,pad=(padding,padding,padding,padding),mode='circular')
            cc=self.conv1(p1)
            cc=self.activ(cc)
#            print ("shape1 ",res1.shape)
            for i in range(self.Nlayer):
                resi=torch.sum(cc,(1,2))
                res1=torch.cat([res1,resi],dim=0)
 #               print ("shape1 in ",i,res1.shape)
                pc=F.pad(cc,pad=(padding,padding,padding,padding),mode='circular')
                cc=self.inconv[i](pc)
                cc=self.activ(cc)
                cc=self.mixlayer[i](cc)
                cc=self.activ(cc)

            cc=self.lastmix(cc)

            
#            print('mipp',cc.shape)
#            summa=self.pool(cc.view(Nx,Ny))
#            print('summashape',summa.shape)
            summa=torch.sum(cc).view(1)
            res1=torch.cat([res1,summa,t_scalar.view(1)],dim=0)
#            print ("shape1 out ",i,res1.shape)
            cc=self.finlin1(res1)
            cc=self.activ(cc)
#            cc=self.finlin2(cc)
#            cc=self.activ(cc)
#            cc=self.finlin3(cc)
#            cc=self.activ(cc)
            cc=self.finlin4(cc).view([])
#            print("ressiz",cc.shape)
            return cc
        R = vmap(net_fn)(x_batch, t_batch)  # (B,)
        grad = vmap(jacrev(net_fn, argnums=0))(x_batch, t_batch)
#        print("rgradsize",R.shape,grad.shape)
        return R, grad
            
        

    def save(self,path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'arch_id': self.__class__.__module__ + "." + self.__class__.__name__,
            'model_config': self.config
        }, path )

    @classmethod    
    def load(cls,path):
        checkpoint=torch.load(path,map_location='cpu')
        model = cls(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
        
        
    def forward_old(self, x_batch, t_batch):
        """
        x_batch: (B, Nx, Ny)
        t_batch: (B,)
        Returns:
            R: (B,)
            grad: (B, Nx, Ny)
        """
        B, Nx, Ny = x_batch.shape

        def net_fn(x_field, t_scalar):
            flat_x = x_field.view(-1)                     # (Nx * Ny,)
            input_vec = torch.cat([flat_x, t_scalar.view(1)], dim=0)  # (Nx * Ny + 1,)
            return self.net(input_vec.unsqueeze(0)).squeeze()         # scalar

        # Compute R and grad using vmap over net_fn
        R = vmap(net_fn)(x_batch, t_batch)  # (B,)
        grad = vmap(jacrev(net_fn, argnums=0))(x_batch, t_batch)  # (B, Nx, Ny)
#        print("shapes",R.shape,grad.shape)
        return R, grad
   
