import numpy as np
from BenchMark_funcs import Five_well_potential_function as target_func
#from BenchMark_funcs import Ackley_function as target_func
#from BenchMark_funcs import Banana as target_func
#from BenchMark_funcs import Eggholder_function as target_func

class Nelder_Mead:
    def __init__(self, num_param, seed, mean=0, sigma=1, simplex_num=None):
        self.num_param = num_param
        self.center = None
        self.P_refl = None
        self.result = None
        self.params = self.initialize_params(seed, mean, sigma, simplex_num)
        
    def initialize_params(self, seed, mean, sigma, simplex_num):
        np.random.seed(seed)
        if simplex_num is None:
            simplex_num = self.num_param+1
        init_params = np.random.normal(loc=mean,
                                       scale=sigma,
                                       size=(simplex_num, self.num_param))
        self.result = self.func(init_params)
        return init_params

    def func(self, inputs):
        y = target_func(inputs)
        result = np.concatenate([inputs, y], axis=-1)  #([p1, f(p1)],[p2, f(p2)], ..., [pm+1, f(pm+1)])
        return result

    def calc(self, alpha=1):
        self.result = self.result[np.argsort(self.result[:, -1])]#sorted on f(px)
        center = np.mean(self.result[:-1, :-1], axis=0)#, keepdims=True)#Calc center
        self.center = center
        center_h = center - self.result[-1, :-1]
        self.center_h = center_h
        refl_coord = center + alpha * (center_h)#[[p_refl]]
        self.P_refl = self.func(refl_coord[None])[0]#[p_refl, f(p_refl)]
        return 0

    def update_opt(self, beta=2, gamma=-0.5, delta=0.5):
        #self.result = self.func(self.params)
        self.calc()

        if self.P_refl[-1] < self.result[-2, -1] and self.P_refl[-1] >= self.result[0, -1]:
            #print("0")
            self.result = np.concatenate([self.result[:-1, :],
                                          self.P_refl[None,:]], axis=0)
            self.params = self.result[..., :-1]
            
        elif self.P_refl[-1] < self.result[0, -1]:
            expand_coord = self.center + beta * self.center_h
            P_expand = self.func(expand_coord[None])[0]
            #print("P_expand:", P_expand)
            if P_expand[-1] < self.P_refl[-1]:
                #print("1-1")
                self.result = np.concatenate([self.result[:-1, :],
                                              self.P_expand[None,:]], axis=0)
                self.params = self.result[..., :-1]
            else:
                #print("1-2")
                self.result = np.concatenate([self.result[:-1, :],
                                              self.P_refl[None,:]], axis=0)
                self.params = self.result[..., :-1]
                
        elif self.P_refl[-1] >= self.result[-2, -1]:
            contract_coord = self.center + gamma * self.center_h
            P_contract = self.func(contract_coord[None])[0]
            #print("P_contract:", P_contract)
            if P_contract[-1] < self.result[-1, -1]:
                #print("2-1")
                self.result = np.concatenate([self.result[:-1, :],
                                              P_contract[None, :]], axis=0)
                self.params = self.result[..., :-1]
                #print(self.params)
            else:
                #print("2-2")
                reduct = (self.result[1:, :-1] + self.result[:1, :-1]) * delta
                #self.result[1:, :self.num_param] + delta * reduct
                self.params = np.concatenate([self.result[:1, :-1],
                                              reduct], axis=0)
                #"""
                tmp_result = self.func(self.params[1:])
                self.result = np.concatenate([self.result[:1, :],
                                              tmp_result], axis=0)
                #"""
                #self.result = self.func(self.params)
                
        else:
            print("[Nelder_Mead]Error...")
            exit()
        #print(self.params)   
        return 0
    
def main(NUM_PARAM, seed, mean, sigma, simplex_num, itera):
    BETA = 2
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    
    lim = 20
    split = 100
    x1 = np.linspace(-lim, lim, split)
    x2 = np.linspace(-lim, lim, split)
    X, Y = np.meshgrid(x1, x2)
    inputs = np.stack([X, Y], axis=-1)

    
    fig = plt.figure()                                                        
    ax = Axes3D(fig)
    ax.plot_wireframe(X,
                      Y,
                      target_func(inputs)[...,0],
                      alpha=0.5)
    
    a = Nelder_Mead(NUM_PARAM, seed, mean, sigma, simplex_num)
    
    print("init_params:", a.params)
    for i in range(itera):
        a.update_opt(beta=BETA)
        """
        ax.scatter(a.center[0],
                   a.center[1],
                   target_func(a.center[None])[...,0],
                   facecolors='red',
                   alpha=1)
        plt.pause(0.1)
        """
    print("Best params:{}, value:{}".format(a.result[0, :-1],
                                            a.result[0, -1]))
    plt.savefig("Trajectory_of_center.png")
    #"""

def scipy_NM(NUM_PARAM, seed, mean, sigma):
    from scipy.optimize import minimize
    #"""
    np.random.seed(seed)
    init_params = np.random.normal(loc=mean,
                                   scale=sigma,
                                   size=(NUM_PARAM+1, NUM_PARAM))
    print("init_params:", init_params)
    res = minimize(target_func, init_params, method='nelder-mead')
    print(res)
    
if __name__ == '__main__':
    #My implimantation
    
    import time
    t = time.time()
    main(NUM_PARAM=2, seed=0, mean=0, sigma=10, simplex_num=6, itera=30)
    print("Time cumume:", time.time() - t)

    """
    #Scipy implimantation

    t = time.time()
    scipy_NM(NUM_PARAM=2, seed=0, mean=0, sigma=100)
    print("Time cumume scipy:", time.time() - t)
    """
