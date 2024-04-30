# gradient descendent code
# goal_1 -> create a function to math function DONE
# goal_2 -> create a function to gradient DONE
# goal_3 -> create a function to algorithm DONE
# goal_4 -> create the graph of results DONE
# goal_5 -> 
import numpy as np
import matplotlib.pyplot as plt

def f(x:float) -> float: 
    return x**2-4*x+1

def grad_f(x:float)->float:
    return 2*x-4

def grad_desc_alg(x:float, eta:float) -> float:
    return x-eta*grad_f(x)

x_values = np.linspace(-5,9,1024)
#x_values = np.linspace(-100,100,1024**2)
u = f(x_values)
discret_x = []
discret_y = []

eta = 0.8
e = 1
x = 9

for i in range(100):
    xn = grad_desc_alg(x,eta)
    e = np.abs(x-xn)
    x = xn
    discret_x += [xn]
    discret_y += [f(xn)]
    plt.clf()
    plt.plot(x_values,u, color = 'black', )
    plt.plot(discret_x,discret_y,marker = 'o', color = 'blue', )
    plt.title(f'Step {i}, Error = {e}')
    plt.grid(True)
    plt.pause(0.5)
    #print(f'step = {i} -> xn = {xn:.3f} e erro = {e:.4f}')
    if e<0.001:
        break    
plt.show()
