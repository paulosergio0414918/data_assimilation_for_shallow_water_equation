import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math
################ variables###############
#the text use large periodic domain [-L,L], where L = 3 
# and number of points N=1024.
L = 3
N = 1024
# The author use T = 2 for observation time
a = 1 
T = 2
# Sample window needs to be greater than 0 end less than 2
Cost = 0
n_testes = 100

############## Initial condition ###########
def u_zero(x:float) -> float: #real and unknown initial condicion
    return (1/20)*np.exp(-(10*x)**2)

def u_g(x:float) -> float: # first guess to initial condition
    return np.zeros(len(x))

##############  Method #########

delta_x = 2*L/N # to evaluate the CFL condition.
M = N/3
delta_t = T/M
CFL = a* delta_t/delta_x

if CFL <= 1:
    print(f'The method could be estable, because CFL = {CFL:.2f}.')

else:
    print(f'The method never be estable, because CFL = {CFL:.2f}.')

kappa1 = 0.5 * (1-CFL)
kappa2 = 0.5 * (1+CFL)

def leap_frog(x):
  u_foreward = np.roll(x,1)
  u_backward = np.roll(x,-1) 
  v = kappa2*u_foreward+kappa1*u_backward
  return v

def leap_frog_transposto(x):
  u_foreward = np.roll(x,1)
  u_backward = np.roll(x,-1) 
  v = kappa1*u_foreward+kappa2*u_backward
  return v

###########collecting samples ################

tempo_das_amostras = [0,0.5,1,1.5]

matriz_de_amostras = np.zeros((N,len(tempo_das_amostras)))

for i in range(N):
  matriz_de_amostras[i,:] = u_zero((i-N/2)*delta_x)


############ Ploting ###############################
x_values = np.linspace(-L,L,N)
d = np.zeros((N,len(tempo_das_amostras)))
gess = u_g(x_values)
u = np.zeros(len(x_values))

error = []


for j in range(n_testes):
    #ploting the graph
    plt.clf()
    plt.ylim(-0.025, 0.1) # y limit
    plt.xlim(-L, L) # x limit
    plt.plot(x_values, gess, lw = 1, color = 'black', )
    plt.plot(x_values, u_zero(x_values), lw = 1, color = 'blue', )
    plt.title(f'Execução {j+1} de {n_testes}.')
    plt.pause(0.1)
    # computing the cost
    Cost1 = 0
    for i in range(N):
        Cost1 = Cost1 + (u_zero(x_values[i])-gess[i])**2
    if Cost1 > Cost:
        print(f'O funcional custo almentou para = {Cost1}.')
    elif Cost1 < Cost:
        print(f'O funcional custo diminuiu para = {Cost1}.')
    else:
        print(f'O funcional custo não alterou')
    error += [Cost1]
    Cost = Cost1
    # computing the discrepance
    for i in range(len(tempo_das_amostras)):
        d[:,i] = gess - matriz_de_amostras[:,i]
    # computing the gradient
    grad = d[:,i]
    for i in range(len(tempo_das_amostras)-1, 0, -1):
        grad = leap_frog_transposto(grad)+d[:,i-1]
    # update the inition condition
    gess = gess-0.3*grad

plt.show()

'''
plt.clf()
vetor = [i for i in range(n_testes)]
plt.plot(vetor, error)
plt.show()
'''


print('End of process.')

#print(error)i