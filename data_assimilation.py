import numpy as np
import matplotlib.pyplot as plt
import random as rd
# things to do
# numerical resolution to advection equation
# study data assimilaton metods

# Initial condition
def u_zero(x:float) -> float: #real and unknow initial condicion
    return (1/20)*np.exp(-(10*x)**2)

def u_g(x:float) -> float: # first guess to initial condition
    denominator = rd.randint(15,25)
    return (1/denominator)*np.exp(-(10*x)**2)

#the text use large periodic domain [-L,L], where L = 3 
# and number of points N=1024.

L = 2
N = 1024

delta_x = 2*L/N # to evaluate the CFL  condition.

x_values = np.linspace(-L,L,1024)

# The author use T = 2 for observation time
a = 1
T = 2
M = N/2

delta_t = T/M

#Lax-Friedrichs Method
CFL = a* delta_t/delta_x
if CFL <= 1:
    print(f'The method could be estable, because CFL = {CFL:.2f}.')

else:
    print(f'The method never be estable, because CFL = {CFL:.2f}.')

#Graphic representation of solution
#u = u_zero(x_values)
u = u_g(x_values)

kappa1 = 0.5 * (1-CFL)
kappa2 = 0.5 * (1+CFL)

Matriz_M = np.zeros([N,N])
for i in range(N):
    for j in range(N):
        if i - j == 1:
            Matriz_M[i,j] = kappa2
        elif i - j == -1:
            Matriz_M[i,j] = kappa1
        elif i == 0 and j == N-1:
            Matriz_M[i,j] = kappa2
        elif j == 0 and i == N-1:
            Matriz_M[i,j] = kappa1



X_amostra = []
Y_amostra = []
for _ in range(10):
    random = rd.uniform(-0.25,0.25)
    X_amostra += [1 + random]
    Y_amostra += [u_zero(random)]


for i in range(450):
    plt.clf()
    plt.ylim(-0.025, 0.1) # y limit
    plt.plot(X_amostra,Y_amostra, marker = 'o', lw = 3, ls = '')
    plt.plot(x_values,u, color = 'black', )
    if i == 256:
        plt.pause(5)
    else:
        plt.pause(0.01)
    u = Matriz_M @ u
    #u_foreward = np.roll(u,1)
    #u_backward = np.roll(u,-1) 
    #u = kappa2*u_foreward+kappa1*u_backward

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

#print(u_zero(0))
a = np.array([0,1,2,3,4,5,6,7,8,9])
b = np.roll(a,1)
c = np.roll(a,-1)
print(a)
print(b)
print(c)


x=[]
y=[]
for _ in range(100):
    x.append(random.randint(0,100))

    y.append(random.randint(0,100))

    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.scatter(x,y, color = 'black')
    plt.pause(0.0001)
plt.show()

'''