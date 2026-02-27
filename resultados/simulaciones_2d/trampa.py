import numpy as np
import jax.numpy as jnp
import jax
from fbpinns.traditional_solutions.seismic_cpml.seismic_CPML_2D_pressure_second_order import seismicCPML2D
from fbpinns.util.logger import logger

class WaveTrampa:
    """Solves the time-dependent (2+1)D wave equation with constant velocity
        d^2 u   d^2 u    1  d^2 u
        ----- + ----- - --- ----- = 0
        dx^2    dy^2    c^2 dt^2

        Boundary conditions:
        u(x,y,0) = amp * exp( -0.5 (||[x,y]-mu||/sd)^2 )
        du
        --(x,y,0) = 0
        dt
    """

    @staticmethod
    def init_params(c0=1, source=np.array([[0., 0., 0.2, 1.]])):

        static_params = {
            "dims":(1,3),
            "c0":c0,
            "c_fn":WaveTrampa.c_fn,# velocity function
            "source":jnp.array(source),# location, width and amplitude of initial gaussian sources (k, 4)
            }
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)),
            (0,(1,1)),
            (0,(2,2)),
        )

        #x_bound = domain.sample_boundaries(all_params, key, sampler, ((0,0),(0,0),(0,0),(0,0),(0,0),(100,100)))
 
        #u_boundary = jnp.zeros((100*100,1))
    
        required_ujs_boundary = (
        (0,()),
    )

        return [[x_batch_phys, required_ujs_phys],] #[x_bound[5], u_boundary, required_ujs_boundary]]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        params = all_params["static"]["problem"]
        c0, source = params["c0"], params["source"]
        x, t = x_batch[:,0:2], x_batch[:,2:3]
        tanh, exp = jax.nn.tanh, jnp.exp

        # get starting wavefield
        p = jnp.expand_dims(source, axis=1)# (k, 1, 4)
        x = jnp.expand_dims(x, axis=0)# (1, n, 2)
        f = (p[:,:,3:4]*exp(-0.5 * ((x-p[:,:,0:2])**2).sum(2, keepdims=True)/(p[:,:,2:3]**2))).sum(0)# (n, 1)

        # form time-decaying anzatz
        t1 = source[:,2].min()/c0
        f = exp(-0.5*(1.5*t/t1)**2) * f
        t = tanh(2.5*t/t1)**2
        return f + t*u

    @staticmethod
    def loss_fn(all_params, constraints):
        c_fn = all_params["static"]["problem"]["c_fn"]
        x_batch, uxx, uyy, utt = constraints[0]
        phys = (uxx + uyy) - (1/c_fn(all_params, x_batch)**2)*utt

        #_, uc, u = constraints[1]

        #boundary_t_end = jnp.mean((u-uc)**2)

        return jnp.mean(phys**2)# + boundary_t_end

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):

        # use the seismicCPML2D FD code with very fine sampling to compute solution

        params = all_params["static"]["problem"]
        c0, source = params["c0"], params["source"]
        c_fn = params["c_fn"]

        (xmin, ymin, tmin), (xmax, ymax, tmax) = np.array(x_batch.min(0)), np.array(x_batch.max(0))

        # get grid spacing
        deltax, deltay, deltat = (xmax-xmin)/(batch_shape[0]-1), (ymax-ymin)/(batch_shape[1]-1), (tmax-tmin)/(batch_shape[2]-1)

        # get f0, target deltas of FD simulation
        f0 = c0/source[:,2].min()# approximate frequency of wave
        DELTAX = DELTAY = 1/(f0*10)*100000# target fine sampled deltas
        DELTAT = DELTAX / (4*np.sqrt(2)*c0)# target fine sampled deltas
        dx, dy, dt = int(np.ceil(deltax/DELTAX)), int(np.ceil(deltay/DELTAY)), int(np.ceil(deltat/DELTAT))# make sure deltas are a multiple of test deltas
        DELTAX, DELTAY, DELTAT = deltax/dx, deltay/dy, deltat/dt
        NX, NY, NSTEPS = batch_shape[0]*dx-(dx-1), batch_shape[1]*dy-(dy-1), batch_shape[2]*dt-(dt-1)

        # get starting wavefield
        xx,yy = np.meshgrid(np.linspace(xmin, xmax, NX), np.linspace(ymin, ymax, NY), indexing="ij")# (NX, NY)
        x = np.stack([xx.ravel(), yy.ravel()], axis=1)# (n, 2)
        exp = np.exp
        p = np.expand_dims(source, axis=1)# (k, 1, 4)
        x = np.expand_dims(x, axis=0)# (1, n, 2)
        f = (p[:,:,3:4]*exp(-0.5 * ((x-p[:,:,0:2])**2).sum(2, keepdims=True)/(p[:,:,2:3]**2))).sum(0)# (n, 1)
        p0 = f.reshape((NX, NY))

        # get velocity model
        x = np.stack([xx.ravel(), yy.ravel()], axis=1)# (n, 2)
        c = np.array(c_fn(all_params, x))
        if c.shape[0]>1: c = c.reshape((NX, NY))
        else: c = c*np.ones_like(xx)

        # add padded CPML boundary
        NPOINTS_PML = 10
        p0 = np.pad(p0, [(NPOINTS_PML,NPOINTS_PML),(NPOINTS_PML,NPOINTS_PML)], mode="edge")
        c =   np.pad(c, [(NPOINTS_PML,NPOINTS_PML),(NPOINTS_PML,NPOINTS_PML)], mode="edge")

        # run simulation
        logger.info(f'Running seismicCPML2D {(NX, NY, NSTEPS)}..')
        wavefields, _ = seismicCPML2D(
                    NX+2*NPOINTS_PML,
                    NY+2*NPOINTS_PML,
                    NSTEPS,
                    DELTAX,
                    DELTAY,
                    DELTAT,
                    NPOINTS_PML,
                    c,
                    np.ones((NX+2*NPOINTS_PML,NY+2*NPOINTS_PML)),
                    (p0.copy(),p0.copy()),
                    f0,
                    np.float32,
                    output_wavefields=True,
                    gather_is=None)

        # get croped, decimated, flattened wavefields
        wavefields = wavefields[:,NPOINTS_PML:-NPOINTS_PML,NPOINTS_PML:-NPOINTS_PML]
        wavefields = wavefields[::dt, ::dx, ::dy]
        wavefields = np.moveaxis(wavefields, 0, -1)
        assert wavefields.shape == batch_shape
        u = wavefields.reshape((-1, 1))

        return u

    @staticmethod
    def c_fn(all_params, x_batch):
        "Computes the velocity model"

        c0 = all_params["static"]["problem"]["c0"]
        return jnp.array([[c0]], dtype=float)# (1,1) scalar value
    
'''def model(self, x, chunk_size=32):
N = x.shape[0]
out = torch.zeros(N, device=x.device)

for i in range(0, self.num_gaussians, chunk_size):
j = min(i + chunk_size, self.num_gaussians)

means = self.means[i:j]
weights = self.weights[i:j]

diff = x[:, None, :] - means[None, :, :]
exponent = torch.einsum(
            "ncd,dd,ncd->nc", diff, self.cov_inv, diff
            )

out += torch.sum(
            torch.exp(-0.5 * exponent) * weights, dim=1
            )

return out
'''