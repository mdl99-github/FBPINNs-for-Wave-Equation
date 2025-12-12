import numpy as np
import jax.numpy as jnp
import jax

# Este script contiene únicamente la clase WaveTrampa() que es idéntica a Wave2D() del módulo fbpinn_wave.
# Sin embargo, dado que se entrenó con este nombre y en un script separado,
# para hacer inferencia con el modelo es necesario que este archivo se encuentre en la misa carpeta.

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
            "c_fn":Wave2D.c_fn,
            "source":jnp.array(source),# location, width and amplitude of initial gaussian sources.
            }
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # Si se usa la versión con puntos de control descomentar las líneas de más abajo y comentar el return actual.

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)),
            (0,(1,1)),
            (0,(2,2)),
        )

        #u_control = jnp.load('u_control.npy')

        #x_control = jnp.load('x_batch.npy')

        required_ujs_control = (
        (0,()),
    )

        #return [[x_batch_phys, required_ujs_phys],[x_control, u_control, required_ujs_control]]
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        params = all_params["static"]["problem"]
        c0, source = params["c0"], params["source"]
        x, t = x_batch[:,0:2], x_batch[:,2:3]
        tanh, exp = jax.nn.tanh, jnp.exp
        
        p = jnp.expand_dims(source, axis=1)
        x = jnp.expand_dims(x, axis=0)
        f = (p[:,:,3:4]*exp(-0.5 * ((x-p[:,:,0:2])**2).sum(2, keepdims=True)/(p[:,:,2:3]**2))).sum(0)

        t1 = source[:,2].min()/c0
        f = exp(-0.5*(1.5*t/t1)**2) * f
        t = tanh(2.5*t/t1)**2
        return f + t*u

    @staticmethod
    def loss_fn(all_params, constraints):
        # Si se usa la versión con puntos de control descomentar las líneas de más abajo.
        c_fn = all_params["static"]["problem"]["c_fn"]
        x_batch, uxx, uyy, utt = constraints[0]
        phys = (uxx + uyy) - (1/c_fn(all_params, x_batch)**2)*utt

        #_, uc, u = constraints[1]

        #boundary = jnp.mean((u - uc)**2)

        return jnp.mean(phys**2) #+ 1e2*boundary

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        # Ignoro la generación de solución exacta
        u = np.random.normal(size=x_batch[:,0].shape).reshape(-1, 1)
        return u

    @staticmethod
    def c_fn(all_params, x_batch):
        "Computes the velocity model"

        c0 = all_params["static"]["problem"]["c0"]
        return jnp.array([[c0]], dtype=float)