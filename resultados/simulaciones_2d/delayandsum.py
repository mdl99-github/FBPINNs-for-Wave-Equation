import numpy as np
from scipy.interpolate import interp1d

##############################################################################
def applyDAS(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,p): #    
    t = np.linspace(to, tf, Nt) # time grid
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) # position of the center of the detectors (3,Ns) [m]
    Pdas = DAS(nx,dx,dsa,posSens,vs,t,p)
    
    return Pdas

###############################################################################
def SensorMaskCartCircleArc(circle_radius, circle_arc, num_sensor_points):
    """
    Matrix with the Ns locations (num_sensor_points) of the sensors arranged 
    on a circunference arc (circle_arc) with a radius circle_radius
    """
    th = np.linspace(0, circle_arc * np.pi / 180, num_sensor_points + 1)
    th = th[0:(len(th) - 1)]  # Angles
    posSens = np.array([circle_radius * np.cos(th), circle_radius * np.sin(th), np.zeros((len(th)))])  # position of the center of the sensors
    posSens = posSens.astype(np.float32)
    return posSens # (3,Ns)

###############################################################################
def createrectgrid2D(nx,dx,zo):
    N = int(nx*nx)
    x = np.linspace(-nx*dx/2,nx*dx/2,nx)
    y = np.linspace(-nx*dx/2,nx*dx/2,nx)
    zv = np.ones((nx,nx))*zo
    xv, yv = np.meshgrid(x,y,indexing='xy')
    rj = np.zeros((3,N)) # pixel position [rj]=(3,N))
    rj[0,:] = xv.ravel()
    rj[1,:] = yv.ravel()
    rj[2,:] = zv.ravel()
    rj = rj.astype(np.float32)
    return rj

###############################################################################
def DAS(nx,dx,dsa,posSens,vs,t,p):
    """
    Traditional Reconstruction Method "Delay and Sum" for 2-D OAT
    The output P0 is the initial pressure [P0] = (N,) where N is the total pixels
    of the image region.
    
    nx: number of pixels in the x direction for a 2-D image region
    dx: pixel size  in the x direction [m]
    dsa: distance sensor array [m]
    posSens: position of the center of the detectors (3,Ns) [m]
    vs: speed of sound (homogeneous medium) [m/s]
    t: time samples (Nt,) [s]
    p: OA measurements (Ns,Nt) where Ns: number of detectors [Pa]
    
    References:
        [1] X. Ma, et.al. "Multiple Delay and Sum with Enveloping Beamforming 
        Algorithm for Photoacoustic Imaging",IEEE Trans. on Medical Imaging (2019).
    """  
    
    # GET Ns, Nt and N
    Ns=p.shape[0]
    Nt=p.shape[1]
    N = nx**2; # Total pixels 
    
    # 2-D IMAGE REGION GRID
    ny = nx  # nx = ny. Rectangular grid
    N = nx * ny  
    #dy = dx  # pixel size in the y direction 
    
    rj = createrectgrid2D(nx,dx,0) # Coordinates of the pixels [3,N]
    rj = rj[0:2,:] # pixel position [rj]=(2,N)
    rj=np.transpose(rj) # (N,2)
    Rj=np.reshape(rj,(1,N*2)) # (1,2*N)
    Rj=np.repeat(Rj,Ns,axis=0) # (Ns,2*N)
    Rj=np.reshape(Rj,(Ns*N*2,1)) # (2*N*Ns,1)
        
    # DETECTOR POSITIONS
    rs=posSens[0:2,:] # (2,Ns)
    rs=np.transpose(rs) # (Ns,2)
    Rs=np.repeat(rs,N,axis=0) # (N*Ns,2)
    Rs=np.reshape(Rs,(Ns*N*2,1)) # (2*N*Ns,1)
 
    # TIME GRID
    Tau=(Rs-Rj)/vs
    Tau=np.reshape(Tau,(Ns*N,2))
    Tau=np.linalg.norm(Tau,ord=2,axis=1) # Get norm 2 by row
    Tau=np.reshape(Tau,(Ns,N))
    
    # OBTAIN 2-D IMAGE
    P0=np.zeros((N,))
    t = np.reshape(t,(1,Nt))
    #for i in tqdm(range(0,Ns)):
    for i in range(0,Ns):
        fp=interp1d(t[0,:],p[i,:]) #interpolación lineal de la mediciones para los tiempo GRILLA
        aux=fp(Tau[i,:])
        P0=P0+aux
    
    #P0 = P0/np.max(P0)
    P0 = np.reshape(P0,(nx,nx)) 
    
    return P0 