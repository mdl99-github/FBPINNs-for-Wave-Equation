import numpy as np
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

from fbpinns.analysis import FBPINN_solution

from scipy.io.matlab import loadmat
import matplotlib.animation as animation
from IPython.display import HTML

class ExactAny:
  '''
    Define una clase para la solución simulada por k-wave de la ecuación

    d^2u/dx^2 - 1/c^2 * d^2u/dt^2 = 0

    Con condiciones de contorno para una f(x) genérica:
    u (x, 0) = f(x)
    u_t (x, 0) = g(x) = 0

  '''
  def __init__(self, file):
    s = loadmat(file)
    self.u = s['exact'][0][0][0]
    self.x = s['exact'][0][0][1]
    self.t = s['exact'][0][0][2][0]
    self.Nx = s['exact'][0][0][3][0][0]
    self.Nt = s['exact'][0][0][4][0][0]
    self.c = s['exact'][0][0][5]

  def plot_colormap(self, **kwargs):
    '''
        Grafica la/las soluciones en un mapa de calor.
    '''
    x, t = np.meshgrid(self.x, self.t, indexing='xy')

    u = self.u[:self.x.shape[0]-1, :self.t.shape[0]-1].T   
    plt.pcolormesh(x, t, u, **kwargs)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Exact Solution ')
    plt.show()

  def animation(self, compare=False, u_compare=None, **kwargs):

    '''
        Grafica la solución n-ésima u_n(x,t) animada a medida que avanza t. Recibe:
        
        'compare': False (Default). Si es True dibuja además de u_n(x,t) la solución recibida en u_compare.

        'u_compare': None (Default). Solución a dibujar en caso de querer comparar con u_n(x,t).
    '''
    fig, ax = plt.subplots()
    line1, = ax.plot(self.x, self.u[:,0], label='k-wave', **kwargs) 
    ax.set_xlabel('Posición en x [m]')
    ax.set_ylabel('Amplitud')
    ax.grid()
    title = ax.set_title('t = 0')
    if compare:
      line2, = ax.plot(self.x, u_compare[:,0], label='FBPINN')
    ax.legend()

    # Función privada de actualización para la animación
    def actualizar(t):
        title.set_text(f't = {self.t[t]:.4f}')
        line1.set_ydata(self.u[:,t])  
        if compare:
          line2.set_ydata(u_compare[:,t])
          return line1, line2, title
        return line1, title


    # Crear la animación
    ani = animation.FuncAnimation(fig, func=actualizar, frames=self.Nt-1, interval=30, blit=True)
    display(HTML(ani.to_jshtml()))
    plt.close(fig)

class ExactC:
  '''
    Define una clase para la solución simulada por k-wave de la ecuación

    d^2u/dx^2 - 1/c^2 * d^2u/dt^2 = 0

    Con condiciones de contorno:
    u (x, 0) = f(x) = e^(-0.5*(x-mu)^2/sigma^2)
    u_t (x, 0) = g(x) = 0

  '''
  def __init__(self, Nx, Nt, xmax, xmin, tmax, tmin, sigma, c):

    self.x = np.linspace(xmin, xmax, Nx)
    self.t = np.linspace(tmin, tmax, Nt)

    x, t = np.meshgrid(self.x, self.t, indexing='xy')
    analytical_sol = lambda x,t: 0.5*jnp.exp(-(x-c*t)**2/2/sigma**2)+0.5*jnp.exp(-(x+c*t)**2/2/sigma**2)

    self.u = analytical_sol(x,t).T
    self.Nx = Nx
    self.Nt = Nt
    self.c = c

  def plot_colormap(self, **kwargs):
    '''
        Grafica la/las soluciones en un mapa de calor.

    '''
    x, t = np.meshgrid(self.x, self.t, indexing='xy')

    u = self.u[:x.shape[0]-1, :x.shape[1]-1].T   
    plt.pcolormesh(x, t, u, **kwargs)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Exact Solution - c = ' + str(self.c))
    plt.show()

  def animation(self, compare=False, u_compare=None, **kwargs):

    '''
        Grafica la solución u(x,t) animada a medida que avanza t. Recibe:

        'compare': False (Default). Si es True dibuja además de u_n(x,t) la solución recibida en u_compare.

        'u_compare': None (Default). Solución a dibujar en caso de querer comparar con u_n(x,t).
    '''
    fig, ax = plt.subplots()
    line1, = ax.plot(self.x, self.u[:,0], label='Analítica', **kwargs) 
    ax.set_xlabel('Posición en x [m]')
    ax.set_ylabel('Amplitud')
    ax.grid()
    title = ax.set_title('t = 0')
    if compare:
      line2, = ax.plot(self.x, u_compare[:,0], label='FBPINN')
    ax.legend()

    # Función privada de actualización para la animación
    def actualizar(t):
        title.set_text(f't = {self.t[t]:.4f}')
        line1.set_ydata(self.u[:,t])  
        if compare:
          line2.set_ydata(u_compare[:,t])
          return line1, line2, title
        return line1, title


    # Crear la animación
    ani = animation.FuncAnimation(fig, func=actualizar, frames=self.Nt-1, interval=30, blit=True)
    display(HTML(ani.to_jshtml()))
    plt.close(fig)

class ExactMix:
  '''
    Define una clase para la solución simulada por k-wave de la ecuación

    d^2u/dx^2 - 1/c^2 * d^2u/dt^2 = 0

    Con condiciones de contorno:
    u (x, 0) = f(x) = sum_i(a_i*e^(-0.5*(x-mu_i)^2/sigma^2))
    u_t (x, 0) = g(x) = 0

  '''
  def __init__(self, file):
    s = loadmat(file)
    self.u = s['exact'][0][0][0]
    self.x = s['exact'][0][0][3]
    self.t = s['exact'][0][0][4][0]
    self.Nx = s['exact'][0][0][5][0][0]
    self.Nt = s['exact'][0][0][6][0][0]
    self.mu = s['exact'][0][0][1]
    self.sigma = s['exact'][0][0][7]
    self.c = s['exact'][0][0][8]
    self.a = s['exact'][0][0][2]

  def plot_colormap(self, all=True, n=0, **kwargs):
    '''
        Grafica la/las soluciones en un mapa de calor. Recibe:

        'all': Si es True (Default) dibuja las soluciones para todos los mu almacenados. Dibuja el n-ésimo en caso contrario.

        'n': 0 (Default). Dibuja la solución con el mu correspondiente a la posición n. 
    '''
    x, t = np.meshgrid(self.x, self.t, indexing='xy')
    if all:
      axn = int(100*len(self.mu) + 10)
      plt.figure("", figsize=(7, 5*len(self.mu)))
      for i in range(0, self.mu.shape[0]):
        plt.subplot(axn + i + 1)
        u = self.u[:-1,:-1,i].T

        plt.pcolormesh(x, t, u, **kwargs)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Exact Solution - mu = ' + str(self.mu[n]) + ' - a = ' + str(self.a[n].round(3)))
    else:
      u = self.u[:x.shape[0]-1, :x.shape[1]-1, n].T   
      plt.pcolormesh(x, t, u, **kwargs)
      plt.colorbar()
      plt.xlabel('x')
      plt.ylabel('t')
      plt.title('Exact Solution - mu = ' + str(self.mu[n]) + ' - a = ' + str(self.a[n].round(3)))
    plt.show()

  def animation(self, n, compare=False, u_compare=None, **kwargs):

    '''
        Grafica la solución n-ésima u_n(x,t) animada a medida que avanza t. Recibe:

        'n': 0 (Default). Dibuja la solución con el mu correspondiente a la posición n. 

        'compare': False (Default). Si es True dibuja además de u_n(x,t) la solución recibida en u_compare.

        'u_compare': None (Default). Solución a dibujar en caso de querer comparar con u_n(x,t).
    '''
    fig, ax = plt.subplots()
    line1, = ax.plot(self.x, self.u[:,0,n], label='k-wave', **kwargs) 
    ax.set_xlabel('Posición en x [m]')
    ax.set_ylabel('Amplitud')
    ax.grid()
    title = ax.set_title('t = 0')
    if compare:
      line2, = ax.plot(self.x, u_compare[:,0], label='FBPINN')
    ax.legend()

    # Función privada de actualización para la animación
    def actualizar(t):
        title.set_text(f't = {self.t[t]:.4f}')
        line1.set_ydata(self.u[:,t,n])  
        if compare:
          line2.set_ydata(u_compare[:,t])
          return line1, line2, title
        return line1, title


    # Crear la animación
    ani = animation.FuncAnimation(fig, func=actualizar, frames=self.Nt-1, interval=30, blit=True)
    display(HTML(ani.to_jshtml()))
    plt.close(fig)

class ExactMulti:
  '''
    Define una clase para la solución simulada por k-wave de la ecuación

    d^2u/dx^2 - 1/c^2 * d^2u/dt^2 = 0

    Con condiciones de contorno:
    u (x, 0) = f(x) = e^(-0.5*(x-mu)^2/sigma^2)
    u_t (x, 0) = g(x) = 0

  '''
  def __init__(self, file):
    s = loadmat(file)
    self.u = s['exact'][0][0][0]
    self.x = s['exact'][0][0][2]
    self.t = s['exact'][0][0][3][0]
    self.Nx = s['exact'][0][0][4][0][0]
    self.Nt = s['exact'][0][0][5][0][0]
    self.mu = s['exact'][0][0][1][0]
    self.sigma = s['exact'][0][0][6]
    self.c = s['exact'][0][0][7]

  def plot_colormap(self, all=True, n=0, **kwargs):
    '''
        Grafica la/las soluciones en un mapa de calor. Recibe:

        'all': Si es True (Default) dibuja las soluciones para todos los mu almacenados. Dibuja el n-ésimo en caso contrario.

        'n': 0 (Default). Dibuja la solución con el mu correspondiente a la posición n. 
    '''
    x, t = np.meshgrid(self.x, self.t, indexing='xy')
    if all:
      axn = int(100*len(self.mu) + 10)
      plt.figure("", figsize=(7, 5*len(self.mu)))
      for i in range(0, len(self.mu)):
        plt.subplot(axn + i + 1)
        u = self.u[:-1,:-1,i].T

        plt.pcolormesh(x, t, u, **kwargs)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Exact Solution - mu = ' + str(self.mu[i]))
    else:
      u = self.u[:x.shape[0]-1, :x.shape[1]-1, n].T   
      plt.pcolormesh(x, t, u, **kwargs)
      plt.colorbar()
      plt.xlabel('x')
      plt.ylabel('t')
      plt.title('Exact Solution - mu = ' + str(self.mu[n]))
    plt.show()

  def animation(self, n, compare=False, u_compare=None, **kwargs):

    '''
        Grafica la solución n-ésima u_n(x,t) animada a medida que avanza t. Recibe:

        'n': 0 (Default). Dibuja la solución con el mu correspondiente a la posición n. 

        'compare': False (Default). Si es True dibuja además de u_n(x,t) la solución recibida en u_compare.

        'u_compare': None (Default). Solución a dibujar en caso de querer comparar con u_n(x,t).
    '''
    fig, ax = plt.subplots()
    line1, = ax.plot(self.x, self.u[:,0,n], label='k-wave', **kwargs) 
    ax.set_xlabel('Posición en x [m]')
    ax.set_ylabel('Amplitud')
    ax.grid()
    title = ax.set_title('t = 0')
    if compare:
      line2, = ax.plot(self.x, u_compare[:,0], label='FBPINN')
    ax.legend()

    # Función privada de actualización para la animación
    def actualizar(t):
        title.set_text(f't = {self.t[t]:.4f}')
        line1.set_ydata(self.u[:,t,n])  
        if compare:
          line2.set_ydata(u_compare[:,t])
          return line1, line2, title
        return line1, title


    # Crear la animación
    ani = animation.FuncAnimation(fig, func=actualizar, frames=self.Nt-1, interval=30, blit=True)
    display(HTML(ani.to_jshtml()))
    plt.close(fig)

class ExactPulse:
  '''
    Define una clase para la solución simulada por k-wave de la ecuación

    d^2u/dx^2 - 1/c^2 * d^2u/dt^2 = 0

    Con condiciones de contorno:
    u (x, 0) = f(x) = e^(-0.5*(x-mu)^2/sigma^2)
    u_t (x, 0) = g(x) = 0

    Para un único mu.

  '''
  def __init__(self, file):
    s = loadmat(file)
    self.u = s['exact'][0][0][0]
    self.x = s['exact'][0][0][1]
    self.t = s['exact'][0][0][2][0]
    self.Nx = s['exact'][0][0][3][0][0]
    self.Nt = s['exact'][0][0][4][0][0]
    self.mu = s['exact'][0][0][5]
    self.sigma = s['exact'][0][0][6]
    self.c = s['exact'][0][0][7]

  def plot_colormap(self, **kwargs):
    '''
        Grafica la/las soluciones en un mapa de calor. 
    '''

    x, t = np.meshgrid(self.x, self.t, indexing='xy')
    u = self.u.T[:x.shape[0]-1, :x.shape[1]-1]   
    plt.pcolormesh(x, t, u, **kwargs)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Exact Solution')
    plt.show()

  def animation(self, compare=False, u_compare=None, **kwargs):\
  
    '''
        Grafica la solución n-ésima u(x,t) animada a medida que avanza t. Recibe: 

        'compare': False (Default). Si es True dibuja además de u(x,t) la solución recibida en u_compare.

        'u_compare': None (Default). Solución a dibujar en caso de querer comparar con u(x,t).
    '''

    # Crear la figura y el eje
    fig, ax = plt.subplots()
    line1, = ax.plot(self.x, self.u[:,0], label='k-wave')  # Inicializar la línea con t=0
    ax.set_xlabel('Posición en x [m]')
    ax.set_ylabel('Amplitud')
    ax.grid()
    title = ax.set_title('t = 0')
    if compare:
      line2, = ax.plot(self.x, u_compare[:,0], label='FBPINN')
    ax.legend()

    # Función de actualización para la animación
    def actualizar(t):
        title.set_text(f't = {self.t[t]:.4f}')
        line1.set_ydata(self.u[:,t])  # Actualizar los datos de la onda
        if compare:
          line2.set_ydata(u_compare[:,t])  # Actualizar los datos de la onda
          return line1, line2, title
        return line1, title


    # Crear la animación
    ani = animation.FuncAnimation(fig, func=actualizar, frames=self.Nt-1, interval=30, blit=True)
    display(HTML(ani.to_jshtml()))
    plt.close(fig)

class ExactSine:
    '''
    Define una clase para la solución simulada por k-wave de la ecuación

    d^2u/dx^2 - 1/c^2 * d^2u/dt^2 = 0

    Con condiciones de contorno:
    u (x, 0) = f(x) = sin(x*pi*L)
    u(0, t) = u(L, t) = 0
    u_t(x, 0) = 0
    '''
    def __init__(self, x, t, u):
        self.x = x
        self.t = t
        self.u = u
        self.Nx = self.x.shape[0]
        self.Nt = self.t.shape[0]

    def plot_colormap(self, **kwargs):
        '''
            Grafica la/las soluciones en un mapa de calor. 
        '''

        x, t = np.meshgrid(self.x, self.t, indexing='xy')
        u = self.u.T[:-1, :-1]   
        plt.pcolormesh(x, t, u, **kwargs)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Exact Solution')
        plt.show()

    def animation(self, compare=False, u_compare=None, **kwargs):\
    
        '''
            Grafica la solución n-ésima u(x,t) animada a medida que avanza t. Recibe: 

            'compare': False (Default). Si es True dibuja además de u(x,t) la solución recibida en u_compare.

            'u_compare': None (Default). Solución a dibujar en caso de querer comparar con u(x,t).
        '''

        # Crear la figura y el eje
        fig, ax = plt.subplots()
        line1, = ax.plot(self.x, self.u[:,0], label='Exacta')  # Inicializar la línea con t=0
        ax.set_xlabel('Posición en x [m]')
        ax.set_ylabel('Amplitud')
        ax.set_xlim([min(self.x), max(self.x)])
        ax.set_ylim([np.min(self.u), np.max(self.u)])
        ax.grid()
        title = ax.set_title('t = 0')
        if compare:
            line2, = ax.plot(self.x, u_compare[:,0], label='FBPINN')
        ax.legend()

        # Función de actualización para la animación
        def actualizar(t):
            title.set_text(f't = {self.t[t]:.4f}')
            line1.set_ydata(self.u[:,t])  # Actualizar los datos de la onda
            if compare:
                line2.set_ydata(u_compare[:,t])  # Actualizar los datos de la onda
                return line1, line2, title
            return line1, title


        # Crear la animación
        ani = animation.FuncAnimation(fig, func=actualizar, frames=self.Nt-1, interval=30, blit=True)
        display(HTML(ani.to_jshtml()))
        plt.close(fig)

    
    

def plot_model_loss(model, const, axis='Epochs', **kwargs):

    '''
        Grafica el error absoluto de la solución del modelo comparando con la exacta. Recibe:

        'model': El modelo definido de acuerdo a la librería de fbpinns. 

        'const': Las constantes correspondientes al modelo de acuerdo con la librería de fbpinns.

        'axis': 'Epochs' (Default). El eje x a graficar. Si es 'Epochs' se grafica el error absoluto a medida que pasan las épocas.
        Si es 'Time', a medida que pasan los minutos de ejecución.
    '''
    assert axis == 'Epochs' or axis == 'Time', 'El eje x pasado no es válido.'

    i, all_params, all_opt_states, active, u_test_losses = model

    plt.figure(figsize=(12,4))
    if axis == 'Epochs':
        plt.plot(u_test_losses[:,0], u_test_losses[:,-1], **kwargs)
        plt.xlabel('Épocas')
    if axis == 'Time':
        plt.plot(u_test_losses[:,3]/60, u_test_losses[:,-1], **kwargs)
        plt.xlabel('Tiempo[mins]')


    plt.yscale("log")
    plt.grid()
    plt.ylabel('Error absoluto')
    plt.title('Evolución del modelo')
    plt.show()

def compare_solution(model, const, sol, multi=False, **kwargs):

    '''
        Grafica todas las suluciones y compara con las exactas en un mapa de calor. Recibe:

        'model': El modelo definido de acuerdo a la librería de fbpinns. 

        'const': Las constantes correspondientes al modelo de acuerdo con la librería de fbpinns.

        'sol': Solución exacta de acuerdo a la clase Exact o ExactMulti.

        'multi': False (Default). Si se va a usar solución con múltiples pulsos o no.

        Devuelve las soluciones generadas.
    '''
   
    axi = 1

    u_todas = []
    ecm_todas = []
    if multi:
        it = sol.mu.shape[0]
        tot_ax = 5
    else:
       it = 1
       tot_ax = 1

    i, all_params, _, active, _ = model

    for i in range(0, it):
        if multi:
            x, t , mu = np.meshgrid(sol.x, sol.t, sol.mu[i], indexing='xy')
            u = FBPINN_solution(const, all_params, active, np.c_[x.flatten(), t.flatten(), mu.flatten() ])
            analytical_sol_results = sol.u[:x.shape[0], :t.shape[0], i].T
        else:
            x, t = np.meshgrid(sol.x, sol.t, indexing='xy')
            u = FBPINN_solution(const, all_params, active, np.c_[x.flatten(), t.flatten()])
            analytical_sol_results = sol.u[:x.shape[0], :t.shape[0]].T

        u = u.reshape(sol.Nt, sol.Nx)

        vmin = min(np.min(analytical_sol_results), np.min(u))
        vmax = max(np.max(analytical_sol_results), np.max(u))

        x = x.reshape(sol.Nt, sol.Nx)
        t = t.reshape(sol.Nt, sol.Nx)

        plt.figure("", figsize=(15, it*4))

        plt.subplot(tot_ax, 3, axi)
        plt.pcolor(x, t, analytical_sol_results[:-1,:-1], vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(r"Solución analítica", fontsize=13)
        plt.xlabel("x", fontsize=13)
        plt.ylabel("t", fontsize=13)

        plt.tight_layout()
        plt.axis("square")

        plt.gca().set_xlim((np.min(sol.x), np.max(sol.x)))
        plt.gca().set_ylim((np.min(sol.t), np.max(sol.t)))

        axi += 1

        plt.subplot(tot_ax, 3, axi)
        plt.pcolormesh(x, t, u, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(r"Solución FBPINN", fontsize=13)
        plt.xlabel("x", fontsize=13)
        plt.ylabel("t", fontsize=13)
        plt.tight_layout()
        plt.axis("square")

        plt.gca().set_xlim((np.min(sol.x), np.max(sol.x)))
        plt.gca().set_ylim((np.min(sol.t), np.max(sol.t)))

        axi += 1

        ecm = np.mean((u - analytical_sol_results)**2)
        ecm_todas.append(ecm)
        ecm = np.format_float_scientific(ecm, precision=3)

        plt.subplot(tot_ax, 3, axi)
        plt.pcolormesh(x, t, (u - analytical_sol_results)**2)
        plt.colorbar()
        plt.title("Error cuadrático - ECM = " + str(ecm), fontsize=13)
        plt.xlabel("x", fontsize=13)
        plt.ylabel("t", fontsize=13)
        plt.tight_layout()
        plt.axis("square")

        plt.gca().set_xlim((np.min(sol.x), np.max(sol.x)))
        plt.gca().set_ylim((np.min(sol.t), np.max(sol.t)))

        axi += 1

        u_todas.append(u)

    ecm_todas = np.array(ecm_todas)
    print("ECM promedio en todas las soluciones: " + str(np.format_float_scientific(np.mean(ecm_todas), precision=3)))
    plt.show()

    return u_todas

def generate_new(model, const, x, t, mu, plot=False, **kwargs):
    
    '''
        Genera la solución para el mu pedido y la grafica. Recibe:

        'model': El modelo definido de acuerdo a la librería de fbpinns. 

        'const': Las constantes correspondientes al modelo de acuerdo con la librería de fbpinns.

        'sol': Solución exacta de acuerdo a la clase Exact o ExactMulti.

        Devuelve la solución u.

    '''

    _, all_params, _, active, _ = model

    Nx = x.shape[0]
    Nt = t.shape[0]

    x, t , mu = np.meshgrid(x, t, np.array([mu]), indexing='xy')
    u = FBPINN_solution(const, all_params, active, np.c_[x.flatten(), t.flatten(), mu.flatten()])


    u = u.reshape(Nt, Nx)
    if plot:
        vmin = np.min(u)
        vmax = np.max(u)

        x = x.reshape(Nx, Nt)
        t = t.reshape(Nx, Nt)

        plt.figure("", figsize=(15, 7))

        plt.pcolormesh(x, t, u, vmin=vmin, vmax=vmax, **kwargs)
        plt.colorbar()
        plt.title(r"Solución en posición arbitraria FBPINN", fontsize=13)
        plt.xlabel("x", fontsize=13)
        plt.ylabel("t", fontsize=13)
        plt.tight_layout()
        plt.axis("square")

        plt.gca().set_xlim((np.min(x[0,:]), np.max(x[0,:])))
        plt.gca().set_ylim((np.min(t[:,0]), np.max(t[:,0])))

    return u

#-----------------------Definición de los problemas para entrenar----------------------------

class Wave1DMultiPulse:
    """z
          d^2 u    1   d^2 u
          ----- - ---  ----- = 0
          dx^2    c^2  dt^2

        Boundary conditions:
        u (x, 0) = f(x) = e^(-0.5*(x-mu)^2/sigma^2)
        u_t (x, 0) = g(x) = 0
    """

    @staticmethod
    def init_params(c, sigma, u_exact):
        static_params = {
            "dims":(1,3),
            "c":c,
            "sigma":sigma,
            "u_exact":u_exact
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)), #uxx
            (0,(1,1)), #utt
        )

        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):

        x, t, c, mu, sigma = x_batch[:,0:1], x_batch[:,1:2], all_params["static"]["problem"]["c"], x_batch[:,2:3], all_params["static"]["problem"]["sigma"]
        t1 = sigma/c
        
        u = jax.nn.sigmoid(5*(2-t/t1))*jnp.exp(-0.5*((x-mu)/sigma)**2) + u*jnp.tanh(t/t1)**2

        return u

    @staticmethod
    def loss_fn(all_params, constraints):

        c = all_params["static"]["problem"]["c"]

        # physics loss
        _, uxx, utt = constraints[0]
        phys = jnp.mean((uxx - utt/c**2)**2)

        return phys

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        return all_params["static"]["problem"]["u_exact"]
    

class Wave1DPulse:
    """z
          d^2 u    1   d^2 u
          ----- - ---  ----- = 0
          dx^2    c^2  dt^2

        Boundary conditions:
        u (x, 0) = f(x) = e^(-0.5*(x-mu)^2/sigma^2)
        u_t (x, 0) = g(x) = 0
    """

    @staticmethod
    def init_params(c, mu, sigma, u_exact):
        static_params = {
            "dims":(1,2),
            "c":c,
            "mu":mu,
            "sigma":sigma,
            "u_exact":u_exact
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        x_bound = domain.sample_boundaries(all_params, key, sampler, ((100,),(100,),(100,),(1,)))
        required_ujs_phys = (
            (0,(0,0)), #uxx
            (0,(1,1)), #utt
        )

        return [[x_batch_phys, required_ujs_phys],]


    @staticmethod
    def constraining_fn(all_params, x_batch, u):

        x, t, c, mu, sigma = x_batch[:,0:1], x_batch[:,1:2], all_params["static"]["problem"]["c"], all_params["static"]["problem"]["mu"], all_params["static"]["problem"]["sigma"]
        t1 = sigma/c
        a = 1

        u = jax.nn.sigmoid(5*(2-t/t1))*jnp.exp(-0.5*((x-mu)/sigma)**2) + u*jnp.tanh(t/t1)**2
        return u

    @staticmethod
    def loss_fn(all_params, constraints):

        c = all_params["static"]["problem"]["c"]

        # physics loss
        _, uxx, utt = constraints[0]
        phys = jnp.mean((uxx - utt/c**2)**2)
        '''
        _, uc, u = constraints[1]
        boundary_1 = jnp.mean((u-uc)**2)

        _, uc, u = constraints[2]
        boundary_2 = jnp.mean((u-uc)**2)
        '''
        return phys #+ boundary_1 + boundary_2

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        return all_params["static"]["problem"]["u_exact"]
    
#Función auxiliar para forzar que se cumplan las condiciones de contorno
def delta(x):
  return (x==0)*1
    
class Wave1D:
    """z
          d^2 u    1   d^2 u
          ----- - ---  ----- = 0
          dx^2    c^2  dt^2

        Boundary conditions:
        u (0,t) = 0
        u (L, t) = 0
        u (x, 0) = f(x) = sin(pi*x)
        u_t (x, 0) = g(x) = 0
    """

    @staticmethod
    def init_params(c=3, L=1):
        '''
        Inicializa los parámetros de la ecuación diferencial y los límites del problema.
        '''
        static_params = {
            "dims":(1,2),
            "c":c,
            "L":L
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        '''
        Muestrea el dominio en cada época. Devuelve además del muestreo, una tupla que indica las derivadas que se van a necesitar para calcular el residual de la ecuación diferencial.
        Por ejemplo (0,()) hace referencia a u y (0,(0,1)) hace referencia a u_xt.
        '''

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        #x_bound = domain.sample_boundaries(all_params, key, sampler, ((100,),(100,),(100,),(1,))) #Por si se usan condiciones soft. Se tienen que devolver también los valores de u y sus derivadas en dicho contorno.
        required_ujs_phys = (
            (0,(0,0)), #u_xx
            (0,(1,1)), #u_tt
        )

        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):

        '''
        Función de que fuerza las condiciones de contorno a la salida. Si u es la salida de la red pura se le aplica u* = constraining_fn(u) de manera que la salida final sea u* y que u* cumpla con las
        condiciones de contorno, sin necesidad de muestrear específicamente allí.
        '''

        x, t, sin, L = x_batch[:,0:1], x_batch[:,1:2], jnp.sin, all_params["static"]["problem"]["L"]

        u = (x-L)*x*t**2*u + sin(jnp.pi*x) - sin(jnp.pi*L)*delta(x-L)
        return u

    @staticmethod
    def loss_fn(all_params, constraints):

        '''
        Función de pérdida. Toma las estimaciones de las derivadas necesarias de acuerdo a sample_constraints() y devuelve el residual de la ecuación. Si no se usan condiciones hard
        se debe agegar aquí la pérdida en los contornos.
        '''

        c = all_params["static"]["problem"]["c"]

        # physics loss
        _, uxx, utt = constraints[0]
        phys = jnp.mean((uxx - utt/c**2)**2)

        return phys

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):

        '''
        Solución exacta que se usa para comparar en testeo.
        '''

        c = all_params["static"]["problem"]["c"]

        # use the burgers_solution code to compute analytical solution
        xmin,xmax = x_batch[:,0].min().item(), x_batch[:,0].max().item()
        tmin,tmax = x_batch[:,1].min().item(), x_batch[:,1].max().item()
        vx = np.linspace(xmin,xmax,batch_shape[0])
        vt = np.linspace(tmin,tmax,batch_shape[1])
        vx,vt = np.meshgrid(vx,vt)

        u = lambda vx,vt: 0.5*jnp.sin(jnp.pi*(vx-c*vt))+0.5*jnp.sin(jnp.pi*(vx+c*vt))
        vu = u(vx,vt).reshape(-1,1, order='F')

        return vu
    
class Wave1DMultiC:
    """z
          d^2 u    1   d^2 u
          ----- - ---  ----- = 0
          dx^2    c^2  dt^2

        Boundary conditions:
        u (x, 0) = f(x) = e^(-0.5*(x/sigma)**2)
        u_t (x, 0) = g(x) = 0
    """

    @staticmethod
    def init_params(sigma, u_exact):
        static_params = {
            "dims":(1,3),
            "sigma":sigma,
            "u_exact":u_exact
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)), #uxx
            (0,(1,1)), #utt
        )

        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):

        x, t, c, sigma = x_batch[:,0:1], x_batch[:,1:2], x_batch[:,2:3], all_params["static"]["problem"]["sigma"]
        t1 = sigma/c
        
        u = jax.nn.sigmoid(5*(2-t/t1))*jnp.exp(-0.5*(x/sigma)**2) + u*jnp.tanh(t/t1)**2

        return u

    @staticmethod
    def loss_fn(all_params, constraints):

        # physics loss
        x_batch_phys, uxx, utt = constraints[0]
        c = x_batch_phys[:,2:3]
        phys = jnp.mean((uxx - utt/c**2)**2)

        return phys

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):

        return all_params["static"]["problem"]["u_exact"]
    
def triang(mu, w, x):
  t = jnp.zeros_like(x)
  m = 2/w
  b1 = (w/2-mu)*m
  b2 = (w/2 + mu)*m
  t = jnp.where(x > mu - w/2, m*x+b1, t)
  t = jnp.where(x > mu, -m*x+b2, t)
  t = jnp.where(x > mu + w/2, 0, t)
  return t

class Wave1DTriang: #soft
    """
          d^2 u    1   d^2 u
          ----- - ---  ----- = 0
          dx^2    c^2  dt^2

        Boundary conditions:
        u (x, 0) = f(x) = triang(0, width, x)
        u_t (x, 0) = g(x) = 0
    """

    @staticmethod
    def init_params(mu, width, c, u_exact):
        static_params = {
            "dims":(1,2),
            "mu":mu,
            "width":width,
            "c":c,
            "u_exact":u_exact
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)), #uxx
            (0,(1,1)), #utt
        )

        # boundary loss
        x_bound = domain.sample_boundaries(all_params, key, sampler, ((1,),(1,),(300,),(1,)))

        width, mu = all_params["static"]["problem"]["width"], all_params["static"]["problem"]["mu"]
        u_boundary = triang(mu, width, x_bound[2][:,0], n_max=None).reshape(300,1)
        ut_boundary = jnp.zeros_like(x_bound[2][:,1]).reshape(300,1)
        required_ujs_boundary = (
        (0,()),
        (0,(1,)),
    )


        return [[x_batch_phys, required_ujs_phys],
                [x_bound[2], u_boundary, required_ujs_boundary], [x_bound[2], ut_boundary, required_ujs_boundary]]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        return u

    @staticmethod
    def loss_fn(all_params, constraints):

        # physics loss
        _, uxx, utt = constraints[0]
        c = all_params["static"]["problem"]["c"]
        phys = jnp.mean((uxx - utt/c**2)**2)

        # boundary loss
        _, uc, u, ut = constraints[1]
        boundary_1 = jnp.mean((u-uc)**2)

        _, utc, u, ut = constraints[2]
        boundary_2 = jnp.mean((ut-utc)**2)

        return phys + boundary_1 + boundary_2

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):

        return all_params["static"]["problem"]["u_exact"]
    
  