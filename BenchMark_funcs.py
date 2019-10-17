import numpy as np

def Five_well_potential_function(inputs):
    x1 = inputs[...,  :1]
    x2 = inputs[..., 1:2]
    #return x1**2 + x2**2
    return (1 - 1 / (1 + 0.05 * (x1**2 + (x2 - 10)**2)) \
    #return (1 - 1 / (1 + 0.05 * (x1**2 + x2 - 10)**2) \
            -1 / (1 + 0.05 * ((x1 - 10)**2 + x2**2)) \
            -1.5 / (1 + 0.03 * ((x1 + 10)**2 + x2**2)) \
            -2 / (1 + 0.05 * ((x1 - 5)**2 + (x2 + 10)**2)) \
            -1 / (1 + 0.1 * ((x1 + 5)**2 + (x2 + 10)**2))) * \
            (1 + 1e-4 * np.power(x1**2 + x2**2, 1.2))
    #"""

def Ackley_function(inputs):
    x1 = inputs[...,  :1]
    x2 = inputs[..., 1:2]
    return - 20 * np.exp(-0.2*np.sqrt((x1**2+x2**2)/2)) - np.exp((np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2))/2) + 20 + np.exp(1)
    #return 20 - 20 * np.exp(-0.2*np.sqrt((x1**2+x2**2)/2)) + np.e - np.exp((np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2))/2)

def Banana(inputs):
    x1 = inputs[...,  :1]
    x2 = inputs[..., 1:2]
    return 100 * (x2 - x1**2)**2 + (x1-1)**2

def Eggholder_function(inputs):
    x1 = inputs[...,  :1]
    x2 = inputs[..., 1:2]
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + 0.5*x1 + 47))) \
        -x1 * np.sin(np.sqrt(np.abs(x1 - x2 - 47)))

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()                                                        
    ax = Axes3D(fig)

    x1 = np.linspace(-20, 20, 100)
    x2 = np.linspace(-20, 20, 100)
    X, Y = np.meshgrid(x1, x2)
    inputs = np.stack([X, Y], axis=-1)
    print(inputs.shape)
    y = Five_well_potential_function(inputs)
    print(y.shape)
    ax.plot_wireframe(X, Y, y[...,0])
    plt.show()

    
