# FBPINNs para la Ecuación de Onda 1D

  

Proyecto con resultados y templates para usar el módulo de FBPINNS[^1] . Sea la ecuación de onda unidimensional:

$$\frac{\partial ^ 2 u}{\partial x^2} - \frac{1}{c ^ 2} \frac{\partial ^ 2u}{\partial t ^ 2} = 0$$

Se pretende encontrar soluciones para distintos valores iniciales $u(x,0)$ y demás condiciones de contorno, incorporando el residual de la ecuación en la pérdida del modelo explícitamente[^3]:

$$\mathcal{L}(x_i,t_i,\mathbf{w}) = \frac{1}{N}\sum_{i=1}^N(\frac{\partial ^ 2 u(x_i,t_i,\mathbf{w})}{\partial x^2} - \frac{1}{c ^ 2} \frac{\partial ^ 2u(x_i,t_i,\mathbf{w})}{\partial t ^ 2})^2$$

Con $u^*(x_i,t_i,\mathbf{w})$ la salida final de la red.

## Contiene:
- Ejemplo resolviendo la ecuación de onda unidimensional para condición inicial senoidal, comentado y que puede ser usado como template para resolver otros problemas.
- Resultados para la condición inicial de pulso gaussiano para distintas situaciones.
- Módulo complementario con algunas funciones útiles que se confeccionaron para facilitar la visualización y comparación de resultados.

## Instalación

Si se quieren todos los archivos se puede utilizar:

```
 git clone https://github.com/mdl99-github/FBPINNs-for-1D-Wave-Equation
```

Si se quiere instalar el módulo `fbpinn_wave` en un entorno de Python:
```
pip install git+https://github.com/mdl99-github/FBPINNs-for-1D-Wave-Equation
```

## Uso

En el archivo `fbpinns_1d_seno.ipynb` se encuentran comentarios sobre cómo usar los modelos. Dado que se usa la arquitectura de FBPINN, es recomendable mirar el repositorio correspondiente[^2] para utilizar notación adecuada.

  

[^1]: [Moseley, B., Markham, A. & Nissen-Meyer, T. Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving differential equations. _Adv Comput Math_ **49**, 62 (2023). https://doi.org/10.1007/s10444-023-10065-9](https://link.springer.com/article/10.1007/s10444-023-10065-9).
[^2]:[FBPINNs Github](https://github.com/benmoseley/FBPINNs)
[^3]:[Hubert Baty. A hands-on introduction to Physics-Informed Neural Networks for solving partial dif-
ferential equations with benchmark tests taken from astrophysics and plasma physics. 2024. ffhal-
04491808ff](https://arxiv.org/abs/2403.00599v1)