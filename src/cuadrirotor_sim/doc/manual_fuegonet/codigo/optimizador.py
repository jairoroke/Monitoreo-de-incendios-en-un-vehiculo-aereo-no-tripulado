"""MuSGD: el optimizador de YOLO26 (Muon + SGD con momento de Nesterov).

Muon toma el momento del gradiente de una matriz de pesos G (una convolución
c_out x c_in x k x k se ve como matriz c_out x (c_in k k)) y la reemplaza por
su «parte ortogonal» U V^T (de G = U S V^T). Así todas las direcciones
singulares avanzan con el mismo tamaño de paso y ninguna domina. U V^T se
aproxima con 5 iteraciones de Newton-Schulz, sin calcular la SVD.

En YOLO26 los pesos de convolución reciben las DOS actualizaciones:
    p <- p - lr * 0.2 * Muon(g)  -  lr * 1.0 * SGD(g + wd p)
y los sesgos y BatchNorm solo la de SGD.
"""
import torch
from torch import nn


def ortogonalizar(G, pasos=5, eps=1e-7):
    """Newton-Schulz quíntico: devuelve aproximadamente U V^T con G = U S V^T."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)                 # valor singular mayor <= 1
    alta = G.size(0) > G.size(1)
    if alta:
        X = X.T
    for _ in range(pasos):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return (X.T if alta else X).to(G.dtype)


class MuSGD(torch.optim.Optimizer):

    def __init__(self, grupos, lr=0.01, momento=0.9, wd=0.0, f_muon=0.2, f_sgd=1.0):
        super().__init__(grupos, dict(lr=lr, momentum=momento, weight_decay=wd, muon=False))
        self.f_muon, self.f_sgd = f_muon, f_sgd

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, mu = g['lr'], g['momentum']
            for p in g['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st['m_sgd'] = torch.zeros_like(p)
                    if g['muon']:
                        st['m_muon'] = torch.zeros_like(p)
                grad = p.grad
                f = 1.0
                if g['muon']:
                    m = st['m_muon'].lerp_(grad, 1 - mu)             # m = mu m + (1-mu) g
                    u = grad.lerp(m, mu)                             # Nesterov
                    u2 = u.view(u.size(0), -1)
                    o = ortogonalizar(u2) * max(1, u2.size(0) / u2.size(1)) ** 0.5
                    p.add_(o.view_as(p), alpha=-lr * self.f_muon)
                    f = self.f_sgd
                d = grad.add(p, alpha=g['weight_decay']) if g['weight_decay'] else grad
                buf = st['m_sgd'].mul_(mu).add_(d)
                p.add_(d.add(buf, alpha=mu), alpha=-lr * f)          # SGD con Nesterov


def grupos_de_parametros(modelo, wd=5e-4, muon=True):
    """Pesos de conv con decaimiento (+Muon), BatchNorm y sesgos sin decaimiento."""
    pesos, bn, sesgos = [], [], []
    for mod in modelo.modules():
        for nombre, p in mod.named_parameters(recurse=False):
            if nombre == 'bias':
                sesgos.append(p)
            elif isinstance(mod, nn.BatchNorm2d):
                bn.append(p)
            else:
                pesos.append(p)
    return [dict(params=pesos, weight_decay=wd, muon=muon),
            dict(params=bn, weight_decay=0.0, muon=False),
            dict(params=sesgos, weight_decay=0.0, muon=False)]
