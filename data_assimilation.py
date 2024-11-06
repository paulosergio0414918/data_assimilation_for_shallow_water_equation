import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import math

############ parâmetros ###################
option = 1  #<-- esta opção apresenta o gráfico da assimilação
#option = 2  #<-- esta opção apresenta o gráfico do custo de assimilação
#option = 3 #<-- refazer os gráficos da figura 4 do artigo kevlahan
#option = 4 #<-- só os graficos da opção 3 
#option = 5 #<--grafico da transfomada de Fourier da condição inicial
#option = 6 #<--grafico da condição inicial
n_testes = 100
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

def fourier(k:float) -> float: #real and unknown initial condicion
    return 1/(200*np.sqrt(2)*np.exp((k**2)/400))


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
  u_forward = np.roll(x,1)
  u_backward = np.roll(x,-1) 
  v = kappa2*u_forward+kappa1*u_backward
  return v

def Lax_Friedrichs_transpose(x):
  u_forward = np.roll(x,1)
  u_backward = np.roll(x,-1) 
  v = kappa1*u_forward+kappa2*u_backward
  return v

def functional_cost(real_solution, approximate_solution):
  if len(real_solution) != len(approximate_solution):
    print('The solutions do not have the same length. Correct this!')
  else:
    cost = 0
    for i in range(len(real_solution)):
      cost += 0.5*(real_solution[i]-approximate_solution[i])**2
  return cost
###########collecting samples ################

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
      plt.ylim(-0.025, 0.06) # y limit
      plt.xlim(-1, 1) # x limit
      plt.plot(x_values, gess, lw = 1, color = 'red', label = '$\phi^{(f)}(x)$' )
      plt.plot(x_values, u_zero(x_values), lw = 1, color = 'blue',label = '$\phi^{(t)}(x)$' )
      plt.title(f'Execução {j+1} de {n_testes} utilizando {n_amostras} amostras com $\Delta x = $ {delta_x}.')
      plt.legend()
      plt.pause(0.1)
      # computing the cost
      Cost1 = n_amostras*functional_cost(u_zero(x_values), gess)
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
  #plt.clf()
  plt.yscale('log')
  plt.xscale('log')
  plt.scatter([i for i in range(n_testes)], error, lw = 1, color = 'blue',label = '$\mathcal{J}(\phi^{(f)}(x))$ em cada iteração' )
  plt.legend()
  plt.show()

#Não mecher aqui
#Não mecher aqui
#Não mecher aqui
elif option == 2:
  norm = []
  error = [1]
  for j in range(n_testes):
      if j == 0:
        local_gess = u_g(x_values)
        Cost1 = n_amostras*functional_cost(u_zero(x_values), gess)
      else:
        error += [n_amostras*functional_cost(u_zero(x_values), gess)/Cost1]
      # computing the discrepance
      for i in range(n_amostras):
          d[:,i] = gess - matriz_de_amostras[:,i]
      # computing the gradient
      grad = d[:,i]
      for i in range(n_amostras-1, 0, -1):
          grad = Lax_Friedrichs_transpose(grad)+d[:,i-1]
      # update the inition condition
      #modificarei apenas daqui
      gamma_local = 1/n_testes
      local_error = 100
      for z in range(n_testes):
        local_gess = local_gess-gamma_local*(z+1)*grad
        local_error1 = functional_cost(u_zero(x_values), local_gess)
        if local_error>local_error1:
           local_error = local_error1
           gamma = gamma_local*(z+1)
      gess = gess-gamma*grad
      local_gess = gess
      #até aqui, de ser merda apage este intervalo e desmarque a linha abaixo
      #gess = gess-0.1*grad
      norm += [np.linalg.norm(u_zero(x_values)-gess)/np.linalg.norm(u_zero(x_values))]
  #plt.ylim(-0.025, error[0]) # y limit
  #plt.xlim(-1, n_testes+1) # x limit
  plt.yscale('log')
  plt.xscale('log')
  plt.xlabel('Iteração $n$')
  plt.plot([i for i in range(n_testes)], error, label = '$\mathcal{J}^{(n)}/\mathcal{J}^{(0)}$')
  plt.plot([i for i in range(n_testes)], norm, color = 'black', label = '$||\phi^{(t)} - \phi^{(n)}||_2/||\phi^{(t)}||_2 $')
  plt.title(f'Gráfico do custo funcional após {n_testes} iterações e {n_amostras} amostras')
  plt.legend()
  plt.show()

#Só aqui
#Só aqui
#Só aqui

elif option == 3:
   Deltax = []
   final_norm = []
   n_amostras_local = 2
   n_testes_local = 750
   distance = 1
   d_local = np.zeros((N,n_amostras_local))
   matriz_de_amostras_local = np.zeros((N,n_amostras_local))
   for i in range(N):
      matriz_de_amostras_local[i,:] = u_zero((i-N/2)*delta_x)
   for c in range(100):
    print(f'Iniciando o processo {c}.')
    Deltax += [(c+1)*delta_x]
    norm = [0]
    error = [1]     
    for j in range(n_testes_local):
          if j == 0:
            local_gess = u_g(x_values)
            local_Cost1 = n_amostras_local*functional_cost(u_zero(x_values), local_gess)
          else:
            error += [n_amostras_local*functional_cost(u_zero(x_values), local_gess)/local_Cost1]
          # computing the discrepance
          for i in range(n_amostras_local):
              d_local[:,i] = np.copy(local_gess - matriz_de_amostras[:,i])
          # computing the gradient
          grad_local = np.copy(d_local[:,1])
          for i in range(n_amostras_local-1, 0, -1):
            for _ in range(distance):
                grad_local = Lax_Friedrichs_transpose(grad_local)
            grad = grad_local+d_local[:,0]
          # update the inition condition
          local_gess = local_gess-0.1*grad
          #modificarei apenas daqui
          #gamma_local = 1/n_testes_local
          #local_error = 100
          #for z in range(n_testes_local):
            #local_gess = local_gess-gamma_local*(z+1)*grad
            #local_error1 = functional_cost(u_zero(x_values), local_gess)
            #if local_error>local_error1:
              #local_error = local_error1
              #gamma = gamma_local*(z+1)
          #gess = gess-gamma*grad
          #local_gess = gess
          #até aqui
          norm += [np.linalg.norm(u_zero(x_values)-local_gess)/np.linalg.norm(u_zero(x_values))]
    distance += 1
    final_norm += [norm[-1]]
    plt.clf()
    plt.xlabel('$\Delta x$')
    plt.ylabel('$||\phi^{(t)} - \phi^{(n)}||_2/||\phi^{(t)}||_2 $')
    #plt.yscale('log')
    #plt.xscale('log')
    #plt.ylim(0.02,0.04) # y limit
    #plt.xlim(-1, n_testes+1) # x limit
    plt.plot(Deltax, final_norm, lw = 1, color = 'blue',label = '$\phi(x)$' )
    plt.title(f'Execução {c+1} de 100.')
    #plt.legend()
    plt.pause(0.1)
   plt.show()
   #print(final_norm) 

if option == 4:
  Deltax = []
  final_norm = []
  n_amostras_local = 2
  n_testes_local = 1000
  distance = 82
  d_local = np.zeros((N,n_amostras_local))
  matriz_de_amostras_local = np.zeros((N,n_amostras_local))
  for i in range(N):
    matriz_de_amostras_local[i,:] = u_zero((i-N/2)*delta_x)
  error_local = []
  distancia = 1
  for j in range(n_testes_local):
      if j == 0:
        local_gess = u_g(x_values)
      else:
        local_gess = local_gess-0.1*grad
      #ploting the graph
      plt.clf()
      plt.ylim(-0.025, 0.06) # y limit
      plt.xlim(-1, 1) # x limit
      plt.xlabel('x')
      plt.plot(x_values, local_gess, lw = 1, color = 'red', label = '$x(t)$' )
      plt.plot(x_values, u_zero(x_values), lw = 1, color = 'blue',label = '$\phi(x)$' )
      plt.title(f'$\Delta x$ = {distance*delta_x}.')
      plt.legend()
      plt.pause(0.1)
      # computing the cost
      Cost1 = n_amostras*functional_cost(u_zero(x_values), local_gess)
      Cost0 = Cost1
      if Cost1 > Cost:
          print(f'Na iteração {j} o funcional custo almentou para = {Cost1}.')
      elif Cost1 < Cost:
          print(f'Na iteração {j} o funcional custo diminuiu para = {Cost1}.')
      else:
          print(f'Na iteração {j} o funcional custo não alterou')
      error_local += [Cost1/Cost0]
      Cost = Cost1
      #final_error = 
      # computing the discrepance
      for i in range(n_amostras_local):
          d_local[:,i] = local_gess - matriz_de_amostras_local[:,i]
      # computing the gradient
      grad_local = d_local[:,1]
      for _ in range(distance):
        grad_local = Lax_Friedrichs_transpose(grad_local)
      grad = Lax_Friedrichs_transpose(grad_local)+d_local[:,0]
      # update the inition condition
      #gess = gess-0.1*grad
  print('Salvar a imagem.')
  plt.pause(30)
  plt.clf()
  plt.yscale('log')
  plt.xscale('log')
  plt.ylabel('$J^{(n)}/J^{(0)}$ em cada iteração')
  plt.scatter([i for i in range(n_testes_local)], error_local, lw = 0.5, color = 'blue',label = '$J^{(n)}/J^{(0)}$ em cada iteração' )
  plt.legend()
  plt.show()     

  
elif option == 5:
  x = [i-40 for i in range(80)]
  y = [fourier(x[i]) for i in range(80)]
  plt.ylim(0, 0.004) # y limit
  plt.xlim(-40, 40) # x limit
  plt.ylabel('E(k)')
  plt.xlabel('Número de onda k')
  plt.plot(x, y, color = 'blue' )
  plt.legend()
  plt.show() 

elif option == 6:
  x = [i-512 for i in range(1024)]
  y = [u_zero(x[i]) for i in range(1024)]
  #print(f'comprimento de y = {len(y)}')
  #print(f'comprimento de x = {len(x)}')
  plt.ylim(0, 0.1) # y limit
  plt.xlim(-3, 3) # x limit
  plt.ylabel('$\phi(x)$')
  plt.xlabel('x')
  plt.plot(x, y, color = 'blue',label = 'Condição inicial verdadeira' )
  plt.legend()
  plt.show()
  
else:
  #print('opção inválida.')
  print('End of process.')


