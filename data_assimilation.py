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
sample_window = [-0.5,0.5] 

n_sample = 10 #number of samples in space
m_sample = 1 #number of samples in time
delta_sample = 0.25 #dispersão das amostras
Cost = 0
############## Initial condition ###########
def u_zero(x:float) -> float: #real and unknown initial condicion
    return (1/20)*np.exp(-(10*x)**2)

def u_g(x:float) -> float: # first guess to initial condition
    denominator = rd.randint(15,25)
    return (1/denominator)*np.exp(-(10*x)**2)

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
'''
Matrix_M = np.zeros([N,N])
for i in range(N):
    for j in range(N):
        if i - j == 1:
            Matrix_M[i,j] = kappa2
        elif i - j == -1:
            Matrix_M[i,j] = kappa1
        elif i == 0 and j == N-1:
            Matrix_M[i,j] = kappa2
        elif j == 0 and i == N-1:
            Matrix_M[i,j] = kappa1
'''

###########collecting samples ################


X_obs = np.linspace(sample_window[0],sample_window[1],n_sample)
Y_obs = u_zero(X_obs)

#Y_obs = np.zeros(len(x_values))



#print(f'Erro = {Cost}.')


################# data assimilation ################

# for i in reversed(range(n)): loop backwards



#grad = u = np.zeros(len(x_values))

#New_u_zero = u + lam * grad


############ Ploting ###############################
x_values = np.array(sorted(set(np.concatenate((np.linspace(-L,L,N),X_obs)))))
#u = u_zero(x_values)
#u = u_g(x_values)
u = np.zeros(len(x_values))
'''
#código antigo que eu estava executando

for j in range(10):
    for i in range(int(math.ceil(N*2.5/(2*L)))):
        plt.clf()
        plt.ylim(-0.025, 0.1) # y limit
        plt.xlim(-0.5, 2.5) # x limit
        plt.plot(X_obs,Y_obs, marker = 'o', ms = 1, ls = '', color = 'red')
        plt.plot(x_values, u, lw = 1, color = 'black', )
        plt.title(f'Execução {j+1} de 10.  T= {i*delta_t*8:.1f} h. de 20h')
        if i == int(math.ceil(M/2)):
            plt.pause(0.01)
            #plt.pause(5)
        else:
            plt.pause(0.01)
            u_foreward = np.roll(u,1)
            u_backward = np.roll(u,-1) 
            u = kappa2*u_foreward+kappa1*u_backward
    Cost1 = sum(0.5*(abs(u_zero(x_values)-u)))#retirei o quadrado aqui
    if Cost1 > Cost:
        print(f'O Erro almentou para = {Cost1}.')
    elif Cost1 < Cost:
        print(f'O Erro diminuiu para = {Cost1}.')
    else:
        print(f'O Erro não alterou')
    Cost = Cost1
    u = 0.5*(u+u_zero(x_values))
print('End of process.')
plt.show()
'''

#nova versão onde vou desconsidera o tempo
for j in range(10):
    plt.clf()
    plt.ylim(-0.025, 0.1) # y limit
    plt.xlim(-0.5, 2.5) # x limit
    plt.plot(X_obs,Y_obs, marker = 'o', ms = 2, ls = '', color = 'red')
    plt.plot(x_values, u, lw = 1, color = 'black', )
    plt.plot(x_values, u_zero(x_values), lw = 1, color = 'blue', )
    plt.title(f'Execução {j+1} de 10.')
    plt.pause(1)
    New_u = 0.5*(u+u_zero(x_values))
    Cost1 = 0
    for j in range(len(X_obs)):
        for i in range(len(x_values)):
            if X_obs[j] == x_values[i]:
                Cost1 += 0.5*(abs(Y_obs[j]-u[i])**2)
                
    if Cost1 > Cost:
        print(f'O funcional custo almentou para = {Cost1}.')
    elif Cost1 < Cost:
        print(f'O funcional custo diminuiu para = {Cost1}.')
    else:
        print(f'O funcional custo não alterou')
    Cost = Cost1
    u = New_u
print('End of process.')
plt.show()








'''

#leap-frog method
#using one step of lax-friederichs to initiate the leapfrog method

u_backward = u_zero(x_values)
u_right = np.roll(u,1)
u_left = np.roll(u,-1) 
u_middle = u = kappa1*u_right+kappa2*u_left
for _ in range(M):
    for i in range(N-1):
        if i == 0 or i == -1:
            u[i] = u_middle[i]
        else:
            u[i]=u_backward[i]+CFL*(u_middle[i+1]-u_middle[i-1])
        u_backward = np.copy(u_middle)
        u_middle = np.copy(u)
    plt.clf()
    plt.plot(x_values,u, color = 'black', )
    plt.pause(0.1)
plt.show()
#Data Assimilation for advection equation

'''