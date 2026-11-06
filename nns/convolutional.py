import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.func import vmap, jacrev
from torch.func import vjp,jvp
from torch.func import grad as torch_grad

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mact(nn.Module):
    def forward(self,x):
        return x * torch.sigmoid(x)
    

class CNet1(nn.Module):
    def __init__(self, Nx, Ny, kernel_size, Nc , Nlayer ):
        super().__init__()
        self.config = {
            'Nx' : Nx ,
            'Ny' : Ny ,
            'kernel_size' : kernel_size,
            'Nc' : Nc ,
            'Nlayer' : Nlayer
        }
        self.Nx = Nx
        self.Ny = Ny
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

        
    def forward(self,x_batch,t_batch):
        B, Nx, Ny = x_batch.shape
        padding=self.padding
        
        def net_fn(x_field,t_scalar):
            bt=t_scalar.expand([Nx,Ny])
            c2t=torch.cat([x_field.view(1,Nx,Ny), bt.view(1,Nx,Ny)], dim=0)
            
            p1=F.pad(c2t,pad=(padding,padding,padding,padding),mode='circular')
            cc=self.conv1(p1)
            cc=self.activ(cc)
            
            for i in range(self.Nlayer):
                pc=F.pad(cc,pad=(padding,padding,padding,padding),mode='circular')
                cc=self.inconv[i](pc)
                cc=self.activ(cc)
                cc=self.mixlayer[i](cc)
                cc=self.activ(cc)

            cc=self.lastmix(cc)
#            print('mipp',cc.shape)
#            summa=self.pool(cc.view(Nx,Ny))
#            print('summashape',summa.shape)
            summa=torch.sum(cc)
            return summa
        R = vmap(net_fn)(x_batch, t_batch)  # (B,)
        grad = vmap(jacrev(net_fn, argnums=0))(x_batch, t_batch)

        return R, grad
            
    def forward_wdiv(self,x_batch,t_batch,n_samples=1,jacord=1,do_test=0):
        B, Nx, Ny = x_batch.shape
        padding=self.padding

        
        def flat_net_fn(flatx,t_scalar):
            squarex=flatx.view(Nx,Ny)
            return net_fn(squarex,t_scalar)


        def get_jac(x_field,t_scalar,fakt):
            flatx=x_field.view([Nx*Ny])
            
            
            jac=jacrev(jacrev(flat_net_fn,argnums=0),argnums=0)(flatx,t_scalar)
#            jac*=fakt
#            jac.diagonal().add_(1.0)
            return jac


        def get_jac_nofakt(x_field,t_scalar):
            flatx=x_field.view([Nx*Ny])
            
            
            jac=jacrev(jacrev(flat_net_fn,argnums=0),argnums=0)(flatx,t_scalar)
            return jac
           
#            aa= torch.log(torch.abs(torch.linalg.det(jac)))
#            return aa
  
        def jactest(x_field,t_scalar,n_samples):
            jac=get_jac(x_field,t_scalar,1.0)
            spur=torch.trace(jac)
            spur2=torch.zeros(1,device=x_field.device,dtype=x_field.dtype)

            grad_g = jacrev(net_fn, argnums=0)

            for _ in range(n_samples):
###                z=torch.randint(0,2,x_field.shape,device=x_field.device)*2-1
                z=torch.randn_like(x_field)
                spur2+=laplacian_hutch3(net_fn,x_field,t_scalar,z)



 

            spur2*=1.0/n_samples

            regispur=spur.item()
            spur2=spur2-spur
            print("spurjoe ",spur2.item(),regispur,n_samples)
            return 

        def jactest2(x_field,t_scalar,n_samples):
            jac=get_jac(x_field,t_scalar,1.0)
            jac2=torch.matmul(jac,jac)
            spur=torch.trace(jac2)
            spur2=torch.zeros(1,device=x_field.device,dtype=x_field.dtype)
            for _ in range(n_samples):
#                z=torch.randint(0,2,x_field.shape,device=x_field.device)*2-1
                z=torch.randn_like(x_field)
                spur2+=laplacian2_hutch3(net_fn,x_field,t_scalar,z)
            spur2*=1.0/n_samples

            regispur=spur.item()
            spur2=spur2-spur
            print("spurjoe2",spur2.item(),regispur,n_samples)
            return 

        def jactest3(x_field,t_scalar,n_samples):
            jac=get_jac(x_field,t_scalar,1.0)
#            jac2=torch.matmul(jac,jac)
            spur=torch.trace(jac)
            spur2=torch.zeros(1,device=x_field.device,dtype=x_field.dtype)
            z=torch.randint(0,2,x_field.shape,device=x_field.device)*2-1
            z=z.to(dtype=x_field.dtype)
#            z=torch.randn_like(x_field)
            spur2+=laplacian_hutch3(net_fn,x_field,t_scalar,z)
          
                            
            spur=z.view([Nx*Ny]) @ jac @ z.view([Nx*Ny])
            regispur=spur
            spur2=spur2-spur
            print("szendvics ",spur2.item(),regispur.item(),n_samples)
            return

        def jactest4(x_field,t_scalar,n_samples):
            jac=get_jac(x_field,t_scalar,1.0)
            jac2=torch.matmul(jac,jac)
            spur=torch.trace(jac)
            spur2=torch.zeros(1,device=x_field.device,dtype=x_field.dtype)

            z=torch.randint(0,2,x_field.shape,device=x_field.device)*2-1
            z=z.to(dtype=x_field.dtype)
#            z=torch.randn_like(x_field)
            spur2+=laplacian2_hutch3(net_fn,x_field,t_scalar,z)
          
                            
            spur=z.view([Nx*Ny]) @ jac2 @ z.view([Nx*Ny])
            regispur=spur
            spur2=spur2-spur
            print("szendvics2 ",spur2.item(),regispur.item(),n_samples)
            return 

        def jactest5(x_field,t_scalar,n_samples,fakt):
            jac=get_jac(x_field,t_scalar,1.0)
            M=jac
            jac=jac*(-fakt)
            jac.diagonal().add_(1.0)
            ddd=torch.log(torch.abs(torch.linalg.det(jac)))
            spur=torch.zeros(1,device=x_field.device,dtype=x_field.dtype)
            spur2=torch.zeros(1,device=x_field.device,dtype=x_field.dtype)

#            grad_g = jacrev(net_fn, argnums=0)

            for _ in range(n_samples):
                z=torch.randint(0,2,x_field.shape,device=x_field.device)*2-1
#                z=torch.randn_like(x_field)
                spur+=laplacian_hutch3(net_fn,x_field,t_scalar,z)
#                spur2+=laplacian2_hutch3(net_fn,x_field,t_scalar,z)
#
#            for _ in range(n_samples):
#                z=torch.randint(0,2,x_field.shape,device=x_field.device)*2-1
#                z=torch.randn_like(x_field)
                spur2+=laplacian2_hutch3(net_fn,x_field,t_scalar,z)                



 

            spur2*=1.0/n_samples
            spur*=1.0/n_samples
 
#            spur=torch.trace(M)
#            spur2=torch.trace(torch.matmul(M,M))
            
            regispur=spur.item()
            newdet=-fakt*spur
            newdet2=-fakt*spur-0.5*fakt*fakt*spur2

            print("detjoe ",(newdet-ddd).item(),(newdet2-ddd).item(),ddd.item(),n_samples)
            return 

        
        
        def net_fn(x_field,t_scalar):
            bt=t_scalar.expand([Nx,Ny])
            c2t=torch.cat([x_field.view(1,Nx,Ny), bt.view(1,Nx,Ny)], dim=0)
            
            p1=F.pad(c2t,pad=(padding,padding,padding,padding),mode='circular')
            cc=self.conv1(p1)
            cc=self.activ(cc)
            
            for i in range(self.Nlayer):
                pc=F.pad(cc,pad=(padding,padding,padding,padding),mode='circular')
                cc=self.inconv[i](pc)
                cc=self.activ(cc)
                cc=self.mixlayer[i](cc)
                cc=self.activ(cc)

            cc=self.lastmix(cc)
#            print('mipp',cc.shape)
#            summa=self.pool(cc.view(Nx,Ny))
#            print('summashape',summa.shape)
            summa=torch.sum(cc)
            return summa

        def laplacian_hutch2(g, x, z ):
                grad_g = jacrev(g,argnums=0)
                z = torch.randint(0, 2, x.shape, device=x.device) * 2 - 1
                _, Hz = jvp(grad_g, (x,), (z,))
                result = result + (z * Hz).sum()

                return (z*Hz).sum()
            
        def laplacian_hutch(g, x, p, z):
            grad_g = jacrev(g, argnums=0)

            _, Hz = jvp(grad_g,(x, p),(z, torch.zeros_like(p)))

            return (z * Hz).sum() 

        def laplacian_hutch3(g, x, p, z):
            grad_g = torch_grad(g, argnums=0)

            _, vjp_fn = vjp(grad_g, x, p)

            Hx_z, _ = vjp_fn(z)

            return (z * Hx_z).sum()

        def laplacian_hutch3_Jsq(g, x, p, z):
            grad_g = torch_grad(g, argnums=0)

            _, vjp_fn = vjp(grad_g, x, p)

            Hx_z, _ = vjp_fn(z)

            rr=(z*Hx_z).sum()
            
            H2x_z, _ = vjp_fn(Hx_z)

            return rr,(z * H2x_z).sum()
        
        def laplacian2_hutch3(g, x, p, z):
            grad_g = torch_grad(g, argnums=0)

            _, vjp_fn = vjp(grad_g, x, p)

            Hx_z, _ = vjp_fn(z)
            
            H2x_z, _ = vjp_fn(Hx_z)

            return (z * H2x_z).sum()


        def laplacian3_hutch3(g, x, p, z):
            grad_g = torch_grad(g, argnums=0)

            _, vjp_fn = vjp(grad_g, x, p)

            Hx_z, _ = vjp_fn(z)            
            H2x_z, _ = vjp_fn(Hx_z)
            H3x_z, _ = vjp_fn(H2x_z)

            return (z * H3x_z).sum()

        
        

        def laplacian_hutch_batch(g, x, p, n_samples):
            lap = 0.0

            for _ in range(n_samples):
                z = torch.randint(
                    0, 2, x.shape,
                    device=x.device,dtype=x.dtype
                ) * 2 - 1

                # vmap only over the batch
                lap += vmap(
                    laplacian_hutch3,
                    in_dims=(None, 0, 0, 0)
                )(g, x, p, z)

            return lap / n_samples
        
        def laplacian_hutch_chunked(g, x, p, n_samples, chunk_size=32):
            B = x.shape[0]
            result = torch.empty(B, device=x.device, dtype=x.dtype)

            lap_single_sample = vmap(
                laplacian_hutch3,
                in_dims=(None, 0, 0, 0)
            )

            for start in range(0, B, chunk_size):
                end = min(start + chunk_size, B)

                xc = x[start:end]
                pc = p[start:end]

                lap = torch.zeros(end - start, device=x.device, dtype=x.dtype)

                for _ in range(n_samples):
                    z = torch.randint(
                        0, 2, xc.shape,
                        device=xc.device
                    ) * 2 - 1

                    lap += lap_single_sample(g, xc, pc, z)

                result[start:end] = lap / n_samples

            return result

        def laplacian2_hutch_chunked(g, x, p, n_samples, chunk_size=32):
            B = x.shape[0]
            result = torch.empty(B, device=x.device, dtype=x.dtype)

            lap_single_sample = vmap(
                laplacian2_hutch3,
                in_dims=(None, 0, 0, 0)
            )

            for start in range(0, B, chunk_size):
                end = min(start + chunk_size, B)

                xc = x[start:end]
                pc = p[start:end]

                lap = torch.zeros(end - start, device=x.device, dtype=x.dtype)

                for _ in range(n_samples):
                    z = torch.randint(
                        0, 2, xc.shape,
                        device=xc.device
                    ) * 2 - 1

                    lap += lap_single_sample(g, xc, pc, z)

                result[start:end] = lap / n_samples

            return result

        def jactraces_hutch_chunked(g, x, p, n_samples, chunk_size=32):
            B = x.shape[0]
            res_tr = torch.empty(B, device=x.device, dtype=x.dtype)
            res_sqtr = torch.empty(B, device=x.device, dtype=x.dtype)
            res_trsq = torch.empty(B, device=x.device, dtype=x.dtype)

            if (n_samples<2):
                print("we need at least 2 noise vectors for Jacobian estimation for 2nd order weight update")
                errorxxxxxxxx

            
            lap_single_sample = vmap(
                laplacian_hutch3_Jsq,
                in_dims=(None, 0, 0, 0)
            )

            for start in range(0, B, chunk_size):
                end = min(start + chunk_size, B)

                xc = x[start:end]
                pc = p[start:end]

                lap = torch.zeros(end - start, device=x.device, dtype=x.dtype)
                lap2 = torch.zeros(end - start, device=x.device, dtype=x.dtype)
                lap3 = torch.zeros(end - start, device=x.device, dtype=x.dtype)

                xxx = torch.zeros(end - start, device=x.device, dtype=x.dtype)
                
                resarr=[]
                
                for _ in range(n_samples):
                    z = torch.randint(
                        0, 2, xc.shape,
                        device=xc.device
                    ) * 2 - 1

                    res,res2=lap_single_sample(g, xc, pc, z)
#                    print("mivanitt ",res.shape,res2.shape)
                    resarr.append(res)
                    
                    lap += res
                    lap2 += res2

#                for i in range(n_samples):
#                    for j in range(i+1,n_samples,1):
#                        lap3+=resarr[i]*resarr[j]

                for i in range(n_samples): 
                    xxx+=resarr[i]*resarr[i]

                        


                    
                res_tr[start:end] = lap / n_samples
                res_trsq[start:end] = lap2 / n_samples
#                res_sqtr[start:end] = lap3 / ( n_samples*(n_samples-1)/2)
                res_sqtr[start:end] =  (xxx/n_samples-lap*lap/(n_samples*n_samples ) ) *( 1.0/(n_samples-1))
 #           print(res_tr,res_trsq,res_sqtr)
            return res_tr,res_trsq,res_sqtr



        
        def jactraces3_hutch_chunked(g, x, p, n_samples, chunk_size=32):
            B = x.shape[0]
            res_tr = torch.empty(B, device=x.device, dtype=x.dtype)
            res_trcub = torch.empty(B, device=x.device, dtype=x.dtype)
            res_trsq = torch.empty(B, device=x.device, dtype=x.dtype)

            lap_single_sample = vmap(
                laplacian_hutch3,
                in_dims=(None, 0, 0, 0)
            )

            lap2_single_sample = vmap(
                laplacian2_hutch3,
                in_dims=(None, 0, 0, 0)
            )

            lap3_single_sample = vmap(
                laplacian3_hutch3,
                in_dims=(None, 0, 0, 0)
            )

            for start in range(0, B, chunk_size):
                end = min(start + chunk_size, B)

                xc = x[start:end]
                pc = p[start:end]

                lap = torch.zeros(end - start, device=x.device, dtype=x.dtype)
                lap2 = torch.zeros(end - start, device=x.device, dtype=x.dtype)
                lap3 = torch.zeros(end - start, device=x.device, dtype=x.dtype)

 #               resarr=[]
                
                for _ in range(n_samples):
                    z = torch.randint(
                        0, 2, xc.shape,
                        device=xc.device
                    ) * 2 - 1

                    res=lap_single_sample(g, xc, pc, z)
                    res2=lap2_single_sample(g, xc, pc, z)
                    res3=lap3_single_sample(g, xc, pc, z)                    
#                    print("mivanitt ",res.shape,res2.shape)
#                    resarr.append(res)
                    
                    lap += res
                    lap2 += res2
                    lap3 += res3

#                for i in range(n_samples):
#                    for j in range(i+1,n_samples,1):
#                        lap3+=resarr[i]*resarr[j]

                    
                res_tr[start:end] = lap / n_samples
                res_trsq[start:end] = lap2 / n_samples
                res_trcub[start:end] =  lap3 / n_samples
 #           print(res_tr,res_trsq,res_sqtr)
            return res_tr,res_trsq,res_trcub
 
        def tracesq(mmm):
            return torch.trace(torch.matmul(mmm,mmm))

        def tracecub(mmm):
            return torch.trace(torch.matmul(torch.matmul(mmm,mmm),mmm))
        
        def jactraces3_exact(x, p, chunk_size=32):
            B = x.shape[0]
            res_tr = torch.empty(B, device=x.device, dtype=x.dtype)
            res_trcub = torch.empty(B, device=x.device, dtype=x.dtype)
            res_trsq = torch.empty(B, device=x.device, dtype=x.dtype)

            jacfunc = vmap( get_jac_nofakt )

            


            for start in range(0, B, chunk_size):
                end = min(start + chunk_size, B)

                xc = x[start:end]
                pc = p[start:end]

 #               resarr=[]

                jac = jacfunc(xc,pc )  
                    
                res_tr[start:end] = vmap(torch.trace)(jac)
                res_trsq[start:end] = vmap(tracesq)(jac)
                res_trcub[start:end] = vmap(tracecub)(jac)
 #           print(res_tr,res_trsq,res_sqtr)
            return res_tr,res_trsq,res_trcub


        def laplacian_hutch_chunked_wsq(g, x, p, n_samples, chunk_size=32):
            B = x.shape[0]
            result = torch.empty(B, device=x.device, dtype=x.dtype)
            result2 = torch.empty(B, device=x.device, dtype=x.dtype)


            
            lap_single_sample = vmap(
                laplacian_hutch3_Jsq,
                in_dims=(None, 0, 0, 0)
            )

            for start in range(0, B, chunk_size):
                end = min(start + chunk_size, B)

                xc = x[start:end]
                pc = p[start:end]

                lap = torch.zeros(end - start, device=x.device, dtype=x.dtype)
                lap2 = torch.zeros(end - start, device=x.device, dtype=x.dtype)

                
                
                for _ in range(n_samples):
                    z = torch.randint(
                        0, 2, xc.shape,
                        device=xc.device
                    ) * 2 - 1

                    res,res2=lap_single_sample(g, xc, pc, z)
                    lap += res
                    lap2 += res2

                result[start:end] = lap / n_samples
                result2[start:end] = lap2 / n_samples

            return result,result2


        

        if do_test:
#            a=net_fn(x_batch[0],t_batch[0])

            jactest5(x_batch[0],t_batch[0],1000,0.01)
            jactest(x_batch[0],t_batch[0],1)
            jactest(x_batch[0],t_batch[0],10)
            jactest(x_batch[0],t_batch[0],10)
            jactest(x_batch[0],t_batch[0],10)
            jactest(x_batch[0],t_batch[0],100)
            jactest(x_batch[0],t_batch[0],640)

            jactest2(x_batch[0],t_batch[0],1)
            jactest2(x_batch[0],t_batch[0],10)
            jactest2(x_batch[0],t_batch[0],100)
            jactest2(x_batch[0],t_batch[0],640)

            jactest3(x_batch[0],t_batch[0],0)
            jactest4(x_batch[0],t_batch[0],0)



        

        
        R = vmap(net_fn)(x_batch, t_batch)  # (B,)
        grad = vmap(jacrev(net_fn, argnums=0))(x_batch, t_batch)

#        logabsdet = vmap( get_ldetjac) (x_batch,t_batch)
#        z = torch.randint(0, 2,(B, n_samples, Nx,Ny), device=x_batch.device,dtype=x_batch.dtype) * 2 - 1

#        f=vmap( 
#            vmap(laplacian_hutch3,in_dims=(None, None, None, 0))
#            ,in_dims=(None,0,0,0)
#            )

 #       laplace=f(net_fn,x_batch,t_batch,z).mean(dim=1)

 #      laplace=laplacian_hutch_batch(net_fn, x_batch, t_batch, n_samples)

        if jacord==2:
#            laplace,laplace2=laplacian_hutch_chunked_wsq(net_fn, x_batch, t_batch, n_samples,1024)
            laplace=laplacian_hutch_chunked(net_fn, x_batch, t_batch, n_samples,1024)
            laplace2=laplacian2_hutch_chunked(net_fn, x_batch, t_batch, n_samples,1024)            
            return R,grad,laplace,laplace2
        
        if jacord==3:
            laplace,laplace2=laplacian_hutch_chunked_wsq(net_fn, x_batch, t_batch, n_samples,1024)
            
            return R,grad,laplace,laplace2

        if jacord==4:
            trM,trMsq,sqtrM = jactraces_hutch_chunked(net_fn, x_batch, t_batch, n_samples,1024)
            return R, grad,trM,trMsq,sqtrM


        if jacord==7:
            trM,trMsq,trMcub = jactraces3_hutch_chunked(net_fn, x_batch, t_batch, n_samples,1024)
            return R, grad,trM,trMsq,trMcub

        if jacord==8:
            trM,trMsq,trMcub = jactraces3_exact( x_batch, t_batch, 32)
            return R, grad,trM,trMsq,trMcub
        

        laplace=laplacian_hutch_chunked(net_fn, x_batch, t_batch, n_samples,1024)
        return R, grad , laplace




    
    def forward_logdetjac(self,x_batch,t_batch,fakt):
        B, Nx, Ny = x_batch.shape
        padding=self.padding

        def flat_net_fn(flatx,t_scalar):
            squarex=flatx.view(Nx,Ny)
            return net_fn(squarex,t_scalar)
        
        def net_fn(x_field,t_scalar):
            bt=t_scalar.expand([Nx,Ny])
            c2t=torch.cat([x_field.view(1,Nx,Ny), bt.view(1,Nx,Ny)], dim=0)
            
            p1=F.pad(c2t,pad=(padding,padding,padding,padding),mode='circular')
            cc=self.conv1(p1)
            cc=self.activ(cc)
            
            for i in range(self.Nlayer):
                pc=F.pad(cc,pad=(padding,padding,padding,padding),mode='circular')
                cc=self.inconv[i](pc)
                cc=self.activ(cc)
                cc=self.mixlayer[i](cc)
                cc=self.activ(cc)

            cc=self.lastmix(cc)
#            print('mipp',cc.shape)
#            summa=self.pool(cc.view(Nx,Ny))
#            print('summashape',summa.shape)
            summa=torch.sum(cc)
            return summa

        
        def get_ldetjac(x_field,t_scalar,fakt):
            flatx=x_field.view([Nx*Ny])
            
            
            jac=jacrev(jacrev(flat_net_fn,argnums=0),argnums=0)(flatx,t_scalar)
            jac*=fakt
            jac.diagonal().add_(1.0)
#            print("jac",jac.shape)
            aa= torch.log(torch.abs(torch.linalg.det(jac)))
#            print("aaaa ",aa.shape)
            return aa

        def get_fullldetjac(x_batch,t_batch,fakt):
            Nb=x_batch.shape[0]
            result=torch.empty(Nb,device=x_batch.device,dtype=x_batch.dtype)

            for i in range(Nb):
                result[i]=get_ldetjac(x_batch[i],t_batch[i],fakt)
            return result

        

        
        R = vmap(net_fn)(x_batch, t_batch)  # (B,)
        grad = vmap(jacrev(net_fn, argnums=0))(x_batch, t_batch)

#        logabsdet = vmap( get_ldetjac,in_dims=(0,0,None) ) (x_batch,t_batch,fakt)
        logabsdet = get_fullldetjac(x_batch,t_batch,fakt)

        
#        z = torch.randint(0, 2,(B, n_samples, Nx,Ny), device=x_batch.device,dtype=x_batch.dtype) * 2 - 1

#        f=vmap( 
#            vmap(laplacian_hutch3,in_dims=(None, None, None, 0))
#            ,in_dims=(None,0,0,0)
#            )

 #       laplace=f(net_fn,x_batch,t_batch,z).mean(dim=1)

 #      laplace=laplacian_hutch_batch(net_fn, x_batch, t_batch, n_samples)
#        laplace=laplacian_hutch_chunked(net_fn, x_batch, t_batch, n_samples,1024)

        return R, grad , logabsdet 
    

    def forward_exactdiv(self,x_batch,t_batch,n_samples=1):
        B, Nx, Ny = x_batch.shape
        padding=self.padding
        
        def net_fn(x_field,t_scalar):
            bt=t_scalar.expand([Nx,Ny])
            c2t=torch.cat([x_field.view(1,Nx,Ny), bt.view(1,Nx,Ny)], dim=0)
            
            p1=F.pad(c2t,pad=(padding,padding,padding,padding),mode='circular')
            cc=self.conv1(p1)
            cc=self.activ(cc)
            
            for i in range(self.Nlayer):
                pc=F.pad(cc,pad=(padding,padding,padding,padding),mode='circular')
                cc=self.inconv[i](pc)
                cc=self.activ(cc)
                cc=self.mixlayer[i](cc)
                cc=self.activ(cc)

            cc=self.lastmix(cc)
#            print('mipp',cc.shape)
#            summa=self.pool(cc.view(Nx,Ny))
#            print('summashape',summa.shape)
            summa=torch.sum(cc)
            return summa
        
        R = vmap(net_fn)(x_batch, t_batch)  # (B,)
        grad = vmap(jacrev(net_fn, argnums=0))(x_batch, t_batch)

        def get_exact_laplace(x_field,t_scalar):
#            print("xmilyen",x_field.shape)
            Nx , Ny = x_field.shape
            div=jacrev(jacrev(net_fn,argnums=0),argnums=0)(x_field,t_scalar)
            dd = 0.0
            for x in range(Nx):
                for y in range(Ny):
                    dd += div[x,y,x,y]
            #print("divmilyen",div.shape)
            # print("dd",dd)
            return dd
 
        laplace = vmap(get_exact_laplace)(x_batch, t_batch)
        
        
        def get_ldetjac(x_field,t_scalar):

            jac=jacrev(jacrev(net_fn,argnums=0),argnums=0)(x_field,t_scalar)
            st_inkabb_legyen_csak_first_order
            return torch.linalg
             
        
        return R, grad , laplace 
    
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
   
