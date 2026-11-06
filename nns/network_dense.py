import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.func import grad,vmap, jacrev

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mact(nn.Module):
    def forward(self,x):
        return x * torch.sigmoid(x)
    

class RNet(nn.Module):
    def __init__(self, Nx, Ny, width):
        super().__init__()
        self.config = {
            'Nx' : Nx ,
            'Ny' : Ny ,
            'width' : width
        }
        self.Nx = Nx
        self.Ny = Ny
        self.width = width
        self.input_dim=Nx*Ny+1
        self.net = nn.Sequential(
            nn.Flatten(),  # (Nx, Ny) -> (Nx*Ny,)
            nn.Linear(self.input_dim , width),
            Mact(),
            nn.Linear(width,width),
            Mact(),
            nn.Linear(width,width),
            Mact(),
            nn.Linear(width,width),
            Mact(),
            nn.Linear(width, 1)  # Scalar output R
        )
        def init_weights(m):
               if isinstance(m, nn.Linear):
          # Example: Xavier uniform initialization
                  nn.init.normal_(m.weight,std=0.105)
                  if m.bias is not None:
                      nn.init.normal_(m.bias,std=0.105)
        self.apply(init_weights) 
                      

            

    def save(self,path,optimizer=None):
        torch.save({
            'model_state_dict': self.state_dict(),
            'arch_id': self.__class__.__module__ + "." + self.__class__.__name__,
#            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': self.config
        }, path )

    @classmethod    
    def load(cls,path,optimizer=None):
        checkpoint=torch.load(path)
        model = cls(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
#        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model
        
        
    def forward(self, x_batch, t_batch):
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
    
    def forward_gradsq(self, x_batch, t_batch):
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

        def get_gradsq(x_field,t_scalar):
            gradvec=jacrev(net_fn,argnums=0)(x_field,t_scalar)
            res=(gradvec*gradvec).sum()

            return res

        
        # Compute R and grad using vmap over net_fn
#        R = vmap(net_fn)(x_batch, t_batch)  # (B,)
#        grad = vmap(jacrev(net_fn, argnums=0))(x_batch, t_batch)  # (B, Nx, Ny)
#        print("shapes",R.shape,grad.shape)
        gradsq= vmap(get_gradsq,in_dims=(0,0))(x_batch,t_batch)
        return gradsq

    
    def laplacian_noisy_single(self,x, t, num_samples=6400):
        Nx, Ny = x.shape
        flat_x = x.view(-1)                     # (Nx * Ny,)
        def net_fn(x_field, t_scalar):
#            print("ittazx-----------------------",x_field)
            flat_x = x_field.view(-1)                     # (Nx * Ny,)
            input_vec = torch.cat([flat_x, t_scalar.view(1)], dim=0)  # (Nx * Ny + 1,)
 #           print("netfben ",x,"sdfsdfsdfsdfsdfffff ",flat_x)
 #           print("ezanet",self.net(input_vec.unsqueeze(0)).squeeze())
            return self.net(input_vec.unsqueeze(0)).squeeze()

        
        D = x.shape[-1]
        result = 0.0
        for _ in range(num_samples):
#           v = torch.randn( flat_x.shape, device=x.device)   # ±1 vector
           v = torch.randint(0, 2, flat_x.shape, device=x.device) * 2 - 1
        #   g = grad(net_fn, argnums=0)(x, t)        # gradient wrt x
           Hv = grad(lambda x_, t: (grad(net_fn, argnums=0)(x_, t) * v).sum(),
                  argnums=0)(flat_x, t)          # Hessian–vector product
           result = result + (Hv * v).sum(-1)            # v^T H v
        return result / num_samples

    def gradtest(self,x, t, num_samples=6400):
        Nx, Ny = x.shape
        flat_x = x.view(-1)                     # (Nx * Ny,)
        def net_fn(x_field, t_scalar):
#            print("ittazx-----------------------",x_field)
            flat_x = x_field.view(-1)                     # (Nx * Ny,)
            input_vec = torch.cat([flat_x, t_scalar.view(1)], dim=0)  # (Nx * Ny + 1,)
 #           print("netfben ",x,"sdfsdfsdfsdfsdfffff ",flat_x)
 #           print("ezanet",self.net(input_vec.unsqueeze(0)).squeeze())
            return self.net(input_vec.unsqueeze(0)).squeeze()

        
        D = x.shape[-1]
        result = 0.0
        Hgrad= grad(net_fn,argnums=0)
        return Hgrad(x,t)

    def laplacian_noisy_single2(self,x, t, num_samples=6400):
        Nx, Ny = x.shape
        flat_x = x.view(-1)                     # (Nx * Ny,)
        def net_fn_flat(x_field, t_scalar):
#            print("ittazx-----------------------",x_field)
            input_vec = torch.cat([x_field, t_scalar.view(1)], dim=0)  # (Nx * Ny + 1,)
 #           print("netfben ",x,"sdfsdfsdfsdfsdfffff ",flat_x)
 #           print("ezanet",self.net(input_vec.unsqueeze(0)).squeeze())
            return self.net(input_vec.unsqueeze(0)).squeeze()

        
        D = x.shape[-1]
        result = 0.0
        for _ in range(num_samples):
#           v = torch.randn( flat_x.shape, device=x.device)   # ±1 vector
           v = torch.randint(0, 2, flat_x.shape, device=x.device) * 2 - 1
           gg = grad(net_fn_flat, argnums=0)        # gradient wrt x
           def gv(x,t):
               return (gg(x,t)*v).sum()
           Hv = grad(gv,argnums=0)(flat_x, t)          # Hessian–vector product
           result = result + (Hv * v).sum()            # v^T H v
        return result / num_samples

    def wasteful_laplacian(self,x,t):
        B, Nx, Ny = x.shape
        batched=vmap(self.wasteful_laplacian_single,in_dims=(0,0))
        return batched(x,t) 

        
    
    def wasteful_laplacian_single(self,x, t):
        Nx, Ny = x.shape
        flat_x = x.view(-1)                     # (Nx * Ny,)
        def net_fn_flat(x_field, t_scalar):
#            print("ittazx-----------------------",x_field)
            input_vec = torch.cat([x_field, t_scalar.view(1)], dim=0)  # (Nx * Ny + 1,)
 #           print("netfben ",x,"sdfsdfsdfsdfsdfffff ",flat_x)
 #           print("ezanet",self.net(input_vec.unsqueeze(0)).squeeze())
            return self.net(input_vec.unsqueeze(0)).squeeze()

        gg = grad(net_fn_flat, argnums=0)

#        print("ggshape ",gg(flat_x,t).shape)
        
        g2 = jacrev(gg,argnums=0)

#        print("gg2shape ",g2(flat_x,t).shape)
        
        result=0.0
        for i in range(Nx*Ny):
            result+=g2(flat_x,t)[i,i]
            
        return result 

    
    

    def lap_disc_single(self,x,t,stepsize):
        Nx, Ny = x.shape
        def net_fn(x_field, t_scalar):
            flat_x = x_field.view(-1)                     # (Nx * Ny,)
            input_vec = torch.cat([flat_x, t_scalar.view(1)], dim=0)  # (Nx * Ny + 1,)
#            return self.net(input_vec.unsqueeze(0)).squeeze()
            return self.net(input_vec.unsqueeze(0)).squeeze()
        res=0.0
        myx=x  # .detach().clone()
        mid=net_fn(myx,t)

        for i in range(Nx):
            for j in range(Ny):
                myx[i,j]+=stepsize
                pp  = net_fn(myx,t)
                myx[i,j]-=2.0*stepsize
                mm  = net_fn(myx,t)
                myx[i,j]+=stepsize
#                mid = net_fn(myx,t)
                res+=mm+pp-mid*2.0
        #        print("itten ",i,j,pp,mm,mid,pp+mm-mid*2.0)
        return res/(stepsize*stepsize)

    def grad_disc_single(self,x,t,stepsize):
        Nx, Ny = x.shape
        def net_fn(x_field, t_scalar):
            flat_x = x_field.view(-1)                     # (Nx * Ny,)
            input_vec = torch.cat([flat_x, t_scalar.view(1)], dim=0)  # (Nx * Ny + 1,)
#            return self.net(input_vec.unsqueeze(0)).squeeze()
            return self.net(input_vec.unsqueeze(0)).squeeze()
        res=x.new_empty([Nx,Ny])

        myx=x.detach().clone()
        mid=net_fn(myx,t)
        for i in range(Nx):
            for j in range(Ny):
                myx[i,j]+=stepsize
                pp  = net_fn(myx,t)
                myx[i,j]-=stepsize
#                mid = net_fn(myx,t)
                res[i,j]=(pp-mid)/stepsize
#                print("itten ",i,j,pp,mid,res[i,j])
#        print("visszares ",res)
        return res
    

    def laplacian_discreet(self,x_batch,t_batch,stepsize):
        B, Nx, Ny = x_batch.shape
        ujx=x_batch.detach().clone()
        res=x_batch.new_empty([B])
        with torch.no_grad():
          for i in range(B):
            res[i]=self.lap_disc_single(ujx[i],t_batch[i],stepsize)
        return res
#        batched=vmap(self.lap_disc_single,in_dims=(0,0,None))
 #       return batched(ujx,t_batch,stepsize)

           


           
    def laplacian_noisy(self,x_batch,t_batch,Nsamp=128):
        B, Nx, Ny = x_batch.shape
        batched_hutch_laplacian = vmap(self.laplacian_noisy_single, in_dims=( 0, 0),randomness="different")
        return batched_hutch_laplacian(x_batch, t_batch,num_samples=Nsamp)

    def t_derivative(self,x_batch,t_batch):
        B, Nx, Ny = x_batch.shape
        def net_fn(x_field, t_scalar):
            flat_x = x_field.view(-1)                     # (Nx * Ny,)
            input_vec = torch.cat([flat_x, t_scalar.view(1)], dim=0)  # (Nx * Ny + 1,)
            return self.net(input_vec.unsqueeze(0)).squeeze()

        t_deriv=grad(net_fn,argnums=1)
        return vmap(t_deriv,in_dims=(0,0))(x_batch,t_batch)


        
    
    def R_laplace(self,x_batch,t_batch):
        B, Nx, Ny = x_batch.shape
        def net_fn(x_field, t_scalar):
##            print("ittazx-----------------------",x_field)
            flat_x = x_field.view(-1)                     # (Nx * Ny,)
            input_vec = torch.cat([flat_x, t_scalar.view(1)], dim=0)  # (Nx * Ny + 1,)
            return self.net(input_vec.unsqueeze(0)).squeeze()

        
        grad_net_fn= grad(net_fn,argnums=0)
        def second_deriv_along_dim(i):
            def g(x,y):
               return grad_net_fn(x,y)[i]
            return grad(g,argnums=0)

        def laplacian_x(x,y):
            hess_diag = vmap(lambda i: second_deriv_along_dim(i)(x, y))(torch.arange(x.shape[0]))
            return hess_diag.sum()


        batched_laplacian_x = vmap(laplacian_x, in_dims=(0, 0))
        lap = batched_laplacian_x(x_batch, t_batch)

        return lap

    def R_laplace_nemjo(self,x_batch,t_batch):
        B, Nx, Ny = x_batch.shape
        def net_fn(x_field, t_scalar):
#            print("ittazx-----------------------",x_field)
            flat_x = x_field.view(-1)                     # (Nx * Ny,)
            input_vec = torch.cat([flat_x, t_scalar.view(1)], dim=0)  # (Nx * Ny + 1,)
            return self.net(input_vec.unsqueeze(0)).squeeze()



    def R_laplace_nemjo(self,x_batch,t_batch):
        B, Nx, Ny = x_batch.shape
        def net_fn(x_field, t_scalar):
#            print("ittazx-----------------------",x_field)
            flat_x = x_field.view(-1)                     # (Nx * Ny,)
            input_vec = torch.cat([flat_x, t_scalar.view(1)], dim=0)  # (Nx * Ny + 1,)
            return self.net(input_vec.unsqueeze(0)).squeeze()
        
        grad_net_fn= grad(net_fn,argnums=0)
        def second_deriv_along_dim(i):
            def g(x,y):
               return grad_net_fn(x,y)[i]
            return grad(g,argnums=0)

        def laplacian_x(x,y):
            hess_diag = vmap(lambda i: second_deriv_along_dim(i)(x, y))(torch.arange(x.shape[0]))
            return hess_diag.sum()


        batched_laplacian_x = vmap(laplacian_x, in_dims=(0, 0))
        lap = batched_laplacian_x(x_batch, t_batch)

        return lap

    
#        hess_diag = vmap(lambda i: second_deriv_along_dim(i)(x_batch,t_batch))(torch.arange(B))
#        return hess_diag.sum()
        
        

    def R_laplace_old(self,x_batch, t_batch):

        B, Nx, Ny = x_batch.shape
        def net_fn(x_field, t_scalar):
#            print("ittazx-----------------------",x_field)
            flat_x = x_field.view(-1)                     # (Nx * Ny,)
            input_vec = torch.cat([flat_x, t_scalar.view(1)], dim=0)  # (Nx * Ny + 1,)
            return self.net(input_vec.unsqueeze(0)).squeeze()         # scalar  
        def per_sample_laplacian(x_field,t_scalar):
#    """
#    f : function mapping (d,) -> scalar
#    x_single : shape (d,), requires_grad=True
#    returns: scalar Laplacian for this one sample
#    """
    # First gradient: (d,)

            grad = torch.autograd.grad(net_fn(x_field,t_scalar) , x_field, create_graph=True)[0]

    # Second derivatives (diagonal only)
            def second_deriv(i):
               g_i = grad[i]
               g2 = torch.autograd.grad(g_i, x_field, retain_graph=True)[0][i]
               return g2

    # Vectorize across dimensions -> (d,)
            return vmap(second_deriv)(torch.arange(x_field.shape[0])).sum()

#    def batched_laplacian(f, x):
#    """
#    f : function mapping (B, d) -> (B,) (scalar per batch element)
#    x : tensor of shape (B, d), requires_grad=True
#  returns: (B,) Laplacian per batch
#     """

#        x_batch.requires_grad=True
#        t_batch.requires_grad=True
        x_batch=x_batch.detach().requires_grad_(True)           
        

    # Wrap per-sample Laplacian, and vmap it across batch
        print("most megy majdxxxxxxxxxxxxxxxxxxxxxxx",x_batch,t_batch)
#        return vmap(lambda x_single: per_sample_laplacian(lambda xx: net_fn(xx.unsqueeze(0))[0], x_single))(x_batch)
        return vmap(per_sample_laplacian)(x_batch,t_batch)




    # --- Example usage ---
#def f(x):
    # x: (B, d)
#    return x[:, 0]**2 + 3 * x[:, 0] * x[:, 1]

#x = torch.tensor([[1.0, 2.0],
 #                 [0.5, -1.0],
 #                 [2.0, 0.0]], requires_grad=True)

#lap = batched_laplacian(f, x)
#print("Laplacians per batch:", lap)
