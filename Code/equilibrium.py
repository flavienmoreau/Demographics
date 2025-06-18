from Code.demographics import Population
from Code.government import Government
from Code.production import Production
import numpy as np


def walras_law(hh, ss, popss: Population, govss: Government, prodss: Production, NFA_Y, r, gamma):
    L = popss.get_L(hh, govss.rho)
    Y_Z = prodss.Y_ZL * L
    g = (1+gamma) * (1 + popss.n) - 1
    I_Y = (g + prodss.delta) * prodss.K_Y
    C_Z, Aj_Z = ss['C'] * popss.N, ss['A_j'] * popss.N # raw hh output in ss is normalized by both Z and N
    NX_Y = 1 - (C_Z / Y_Z + I_Y + govss.G_Y)
    CA_Y = (r - g)*NFA_Y
    MT_Y = popss.migrants() @ Aj_Z / Y_Z * (1+g)
    return NX_Y + CA_Y + MT_Y


def calculate_NFAY_and_Y(ss, hh, pop, gov, prod):
    #B_Y = gov.B_Y
    B_Y = gov.B_Y if np.isscalar(gov.B_Y) else gov.B_Y[:-1]
    K_Y = prod.K_Y
    L = pop.get_L(hh, gov.rho)
    W_Y = ss['W'] / (prod.Y_ZL * L / pop.N)
    Y = prod.Y_ZL * L * prod.Z
    return W_Y - B_Y - K_Y, Y


def get_WY(ss, hh, pop, gov, prod):
    # integrate into above
    L = pop.get_L(hh, gov.rho)
    W_Y = ss['W'] / (prod.Y_ZL * L / pop.N)
    return W_Y


def calculate_aggregates(ss, hh, pop, gov, prod):
    B_Y = gov.B_Y if np.isscalar(gov.B_Y) else gov.B_Y[:-1]
    K_Y = prod.K_Y
    L = pop.get_L(hh, gov.rho)
    W_Y = ss['W'] / (prod.Y_ZL * L / pop.N)
    Y = prod.Y_ZL * L * prod.Z
    NFA_Y = W_Y - B_Y - K_Y
    return dict(W_Y=W_Y, K_Y=K_Y, B_Y=B_Y, NFA_Y=NFA_Y, Y=Y)
