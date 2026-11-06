
def action_2d(phi):
  temp=0.0
  temp2=0.0
  temp3=0.0
  if len(phi.shape)!=2:
      print("not 2 dim")
      exit
      wvvwvwvw
  Nx=phi.shape[0]
  Ny=phi.shape[1]
  for x in range(phi.shape[0]):
    for y in range(phi.shape[1]):
        f=phi[x,y]
        temp+=f*f
        temp2+=f*f*f*f
        temp3+= f* ( phi[(x+1) % Nx ,y]  + phi[x,(y+1) %Ny] )
  return temp*(1.0-2.0*sklambda)+temp2*sklambda  - temp3*2.0*kappa

def force_2dsk_add(phi,force,fakt):
  dinpard=-2.0*(1.0-2.0*sklambda)
  dinparl=-4.0*sklambda
  dinparn=2.0*kappa
  Nx=phi.shape[0]
  Ny=phi.shape[1]
  for x in range(Nx):
    for y in range(Ny):
      f=phi[x,y]
      
      res=(dinpard+dinparl*f*f)*f

      res+= ( phi[ (x+1) % Nx , y]
              +phi[ (x+Nx-1) % Nx , y]
              +phi[ x,(y+1) % Ny]
              +phi[ x,(y+Ny-1) % Ny] )*dinparn

      force[x,y]+=res*fakt

  
def phisq(phi):
  temp=0.0
  Nx=phi.shape[0]
  Ny=phi.shape[1]
  for x in range(Nx):
    for y in range(Ny):
      temp+=phi[x,y]**2
  return temp/(Nx*Ny)

def hmc_trajectory(phi,deltat):
   Nx=phi.shape[0]
   Ny=phi.shape[1]
   mom=np.random.randn(Nx,Ny)
   phic=copy.deepcopy(phi)
   elotte=action_2d(phi)+phisq(mom)*0.5*Nx*Ny
   nstep=int(1.0/deltat)
   for i in range(nstep):
     force_2dsk_add(phi,mom,deltat*0.5)
     phi+=mom*deltat;
     force_2dsk_add(phi,mom,deltat*0.5)
#     print(action_2d(phi)+phisq(mom)*0.5*Nx*Ny-elotte)
   last_action=action_2d(phi)+phisq(mom)*0.5*Nx*Ny
   chance=np.exp(elotte-last_action)
#   print("esely ",chance)
   
   if ( np.random.rand() > chance):
#     print("reject ",phi[0,0],phic[0,0])
     for x in range(Nx):
       for y in range(Ny):
         phi[x,y]=phic[x,y]
#     print("rutan",phi[0,0])
#     reject+=1
   else:
     aa=1
#     accept+=1
