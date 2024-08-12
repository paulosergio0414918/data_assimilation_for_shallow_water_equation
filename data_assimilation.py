import numpy as np
import matplotlib.pyplot as plt
import math

############ parâmetros ###################
#option = 1  #<-- esta opção apresenta o gráfico da assimilação
option = 2  #<-- esta opção apresenta o gráfico do custo de assimilação
#option = 3 #<--(Contém erro) refazer os gráficos da figura 4 do artigo kevlahan
n_testes = 1000
n_amostras = 2

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

def Lax_Friedrichs(x):
  u_foreward = np.roll(x,1)
  u_backward = np.roll(x,-1) 
  v = kappa2*u_foreward+kappa1*u_backward
  return v

def Lax_Friedrichs_transpose(x):
  u_foreward = np.roll(x,1)
  u_backward = np.roll(x,-1) 
  v = kappa1*u_foreward+kappa2*u_backward
  return v

def functional_cost(real_solution, approximate_solution):
  if len(real_solution) != len(approximate_solution):
    print('The solutions not have the same length. Correct this!')
  else:
    cost = 0
    for i in range(len(real_solution)):
      cost += 0.5*(u_zero(x_values[i])-gess[i])**2
  return cost
###########collecting samples ################

#tempo_das_amostras = [0,0.5,1,1.5]

matriz_de_amostras = np.zeros((N,n_amostras))

for i in range(N):
  matriz_de_amostras[i,:] = u_zero((i-N/2)*delta_x)


############ Ploting ###############################

x_values = np.linspace(-L,L,N)
d = np.zeros((N,n_amostras))
gess = u_g(x_values)


if option == 1:
  error = []
  for j in range(n_testes):
      #ploting the graph
      plt.clf()
      plt.ylim(-0.025, 0.1) # y limit
      plt.xlim(-L, L) # x limit
      plt.plot(x_values, gess, lw = 1, color = 'black', )
      plt.plot(x_values, u_zero(x_values), lw = 1, color = 'blue', )
      plt.title(f'Execução {j+1} de {n_testes} utilizando {n_amostras} amostras.')
      plt.pause(0.1)
      # computing the cost
      Cost1 = functional_cost(u_zero(x_values), gess)
      if Cost1 > Cost:
          print(f'O funcional custo almentou para = {Cost1}.')
      elif Cost1 < Cost:
          print(f'O funcional custo diminuiu para = {Cost1}.')
      else:
          print(f'O funcional custo não alterou')
      error += [Cost1]
      Cost = Cost1
      # computing the discrepance
      for i in range(n_amostras):
          d[:,i] = gess - matriz_de_amostras[:,i]
      # computing the gradient
      grad = d[:,i]
      for i in range(n_amostras-1, 0, -1):
          grad = Lax_Friedrichs_transpose(grad)+d[:,i-1]
      # update the inition condition
      gess = gess-0.1*grad

  plt.show()

elif option == 2:
  norm = []
  error = [1]
  for j in range(n_testes):
      if j == 0:
        Cost1 = functional_cost(u_zero(x_values), gess)
      else:
        error += [functional_cost(u_zero(x_values), gess)/Cost1]
      # computing the discrepance
      for i in range(n_amostras):
          d[:,i] = gess - matriz_de_amostras[:,i]
      # computing the gradient
      grad = d[:,i]
      for i in range(n_amostras-1, 0, -1):
          grad = Lax_Friedrichs_transpose(grad)+d[:,i-1]
      # update the inition condition
      gess = gess-0.1*grad
      norm += [np.linalg.norm(u_zero(x_values)-gess)/np.linalg.norm(u_zero(x_values))]
  #plt.ylim(-0.025, error[0]) # y limit
  #plt.xlim(-1, n_testes+1) # x limit
  plt.yscale('log')
  plt.xscale('log')
  plt.xlabel('Iteration $n$')
  plt.plot([i for i in range(n_testes)], error, label = '$\mathcal{J}^{(n)}/\mathcal{J}^{(0)}$')
  plt.plot([i for i in range(n_testes)], norm, color = 'black', label = '$||\phi^{(t)} - \phi^{(n)}||_ 2/||\phi^{(t)}||_2 $')
  plt.title(f'Gráfico do custo funcional após {n_testes} iterações e {n_amostras} amostras')
  plt.legend()
  plt.show()

else:
  n_testes = 50
  error_matrix = np.zeros((n_testes,5))
  norm_matrix = np.zeros((n_testes,5))
  for k in range(5):
    n_amostras = k+2
    d = np.zeros((N,n_amostras))
    matriz_de_amostras = np.zeros((N,n_amostras))
    for b in range(N):
      matriz_de_amostras[b,:] = u_zero((b-N/2)*delta_x)
    norm = [] #second graphics
    error = [1] #first graphics
    for j in range(n_testes):
        if j == 0:
          Cost1 = functional_cost(u_zero(x_values), gess)
        else:
          error += [functional_cost(u_zero(x_values), gess)/Cost1]
        # computing the discrepance
        for i in range(n_amostras):
            d[:,i] = gess - matriz_de_amostras[:,i]
        # computing the gradient
        grad = d[:,i]
        for i in range(n_amostras-1, 0, -1):
            grad = Lax_Friedrichs_transpose(grad)+d[:,i-1]
        # update the inition condition
        if j < 200:
           alpha = 0.3
        elif j>= 200 and j<500:
           alpha = 0.1
        else:
           alpha = 0.09
        gess = gess-0.1*grad
        norm += [np.linalg.norm(u_zero(x_values)-gess)/np.linalg.norm(u_zero(x_values))]
        #print(error)
    error_matrix[:,k] = np.copy(error)
    norm_matrix[:,k] = np.copy(norm)
  #print(error_matrix[:,4])

  #plt.ylim(-0.025, error[0]) # y limit
  #plt.xlim(-1, n_testes+1) # x limit
  plt.yscale('log')
  plt.xscale('log')
  plt.xlabel('Iteration $n$')
  plt.plot([i for i in range(n_testes)], norm_matrix[:,0], label = 'N_obs = 2')
  plt.plot([i for i in range(n_testes)], norm_matrix[:,1], label = 'N_obs = 3')
  plt.plot([i for i in range(n_testes)], norm_matrix[:,2], label = 'N_obs = 4')
  plt.plot([i for i in range(n_testes)], norm_matrix[:,3], label = 'N_obs = 5')
  plt.plot([i for i in range(n_testes)], norm_matrix[:,4], label = 'N_obs = 6')
  plt.title(f'Gráfico do custo funcional após {n_testes} iterações.')
  plt.legend()
  plt.show()

print('End of process.')

