import numpy as np

class Production:
    """Lighter-weight class that will supersede ProductionParam"""
    def __init__(self, alpha, Y_ZL, w_Z, K_Y, Z, delta):
        self.alpha = alpha
        self.Y_ZL = Y_ZL
        self.w_Z = w_Z
        self.K_Y = K_Y
        self.delta = delta
        self.Z = Z
        self.Y_L = Y_ZL * Z

    @staticmethod
    def calibrate(K_Y, Y_L, r, delta):
        alpha = K_Y*(r + delta)
        Z = Y_L / (K_Y)**(alpha/(1-alpha))
        Y_ZL = Y_L / Z
        assert np.isclose(Y_ZL, K_Y**(alpha/(1-alpha)))
        w_Z = (1-alpha) * Y_ZL
        return Production(alpha, Y_ZL, w_Z, K_Y, Z, delta)
    
    def adjust_r(self, r):
        # adjust r and recompute Y_ZL, w_Z, K_Y, Y holding primitives alpha, Z, delta constant
        K_Y = self.alpha / (r + self.delta)
        Y_ZL = (K_Y)**(self.alpha/(1-self.alpha))
        w_Z = (1-self.alpha) * Y_ZL
        return Production(self.alpha, Y_ZL, w_Z, K_Y, self.Z, self.delta)
