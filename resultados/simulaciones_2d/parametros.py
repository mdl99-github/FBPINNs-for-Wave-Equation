import os

import time

from PIL import Image

import torch

import numpy as np
from scipy.stats import multivariate_normal

from jax import jit
from jax import numpy as jnp
from matplotlib import pyplot as plt

from scipy import ndimage

from PIL import Image

class GaussianPulseOptimizer:
    def __init__(self, num_gaussians, cov_matrix, device='cuda'):
        self.num_gaussians = num_gaussians
        self.device = device
        
        self.means = torch.nn.Parameter(torch.empty(num_gaussians, 2, device=device).uniform_(-0.2, 0.2))

        self.weights = torch.nn.Parameter(torch.rand(num_gaussians, device=device))

        
        self.cov_matrix = torch.tensor(cov_matrix, device=device, dtype=torch.float32)
        self.cov_inv = torch.inverse(self.cov_matrix)

    def gaussian_kernel(self, x, mean):
        diff = x - mean
        return torch.exp(-0.5 * torch.sum(torch.matmul(diff, self.cov_inv) * diff, dim=1))

    def model(self, x):
        pulses = torch.stack([self.weights[i] * self.gaussian_kernel(x, self.means[i], self.cov_matrix) 
                              for i in range(self.num_gaussians)], dim=1)
        return torch.sum(pulses, dim=1)

    def fit(self, x, y, num_epochs=1000, lr=0.001):
        optimizer = torch.optim.Adam([self.means, self.weights], lr=lr)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            y_pred = self.model(x)
            
            loss = 1e6*loss_fn(y_pred, y)
            
            loss.backward()
            optimizer.step()

            if epoch == 250:
                optimizer = torch.optim.Adam([self.means, self.weights], lr=0.001)
            if loss <= 1400:
                optimizer = torch.optim.Adam([self.means, self.weights], lr=0.0003)
            #if loss <= 1200:
                #optimizer = torch.optim.Adam([self.means, self.weights], lr=0.0001)

            if loss < 1000:
                optimizer = torch.optim.Adam([self.means, self.weights], lr=0.00001)

            if epoch % 250 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        return self.means, self.weights

img = Image.open('tejido_pequeno2.png') 

im = -(np.array(img.convert('L'))[:,8:-14]/255 - 1)

dx = 15e-3/im.shape[0]

start = time.time()

norm = 7.5e-3

X, Y = torch.meshgrid(
    torch.linspace(-7.5e-3/norm, 7.5e-3/norm, im.shape[0], device='cuda'),
    torch.linspace(-7.5e-3/norm, 7.5e-3/norm, im.shape[1], device='cuda'),
    indexing='xy'
)

x = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


im = torch.Tensor(im).flatten().to('cuda')

sigma2 = (1.5e-3/20/norm)**2
cov_matrix = [[sigma2,0.],[0.,sigma2]]
optimizer = GaussianPulseOptimizer(num_gaussians=800, cov_matrix=cov_matrix, norm=norm)
means, weights = optimizer.fit(x, im, num_epochs=5000, lr=0.05)

end = time.time()

print(f"Tardó {(end-start)/60} minutos")

np.save("w_real2.npy",weights.cpu().detach().numpy())
np.save("mu_real2.npy",means.cpu().detach().numpy())

def gaussian_reconstruction(x, y, w, mu, cov):
    z = np.zeros_like(x)
    for i in range(0, len(w)):
        sigma = np.sqrt(cov[0][0])
        gauss = multivariate_normal(mean=mu[i], cov=cov)
        points = np.stack([x.ravel(), y.ravel()], axis=1)
        pdf = gauss.pdf(points).reshape(x.shape)
        z += w[i]*pdf*np.pi*sigma**2*2
    return z

w = weights.cpu().detach().numpy()
mu = means.cpu().detach().numpy()
cov_matrix = [[sigma2,0.],[0.,sigma2]]

xx = X.cpu().detach().numpy()
yy = Y.cpu().detach().numpy()

u = gaussian_reconstruction(xx,yy, w, mu, cov_matrix)

ec = (u - im.cpu().detach().numpy().reshape(xx.shape))**2
ecm = np.format_float_scientific(np.mean(ec), precision=3)
print(f"El ECM es {ecm}.")

