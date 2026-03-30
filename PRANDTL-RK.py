# Resolução da equação de Prandtl pelo método de Runge Kutta
# Autora: Mylena Carvalho Silva
# Atividade feita para matéria de mecânica dos fludios como trabalho final

import numpy as np
import matplotlib.pyplot as plt
# Parâmetros da Equação de Blasius
U=1
L = 10
mu = 1.8465E-5 #kg/(m·s) (viscosidade dinâmica do ar a 25 °C)
rho = 1.184 #kg/m³ (densidade do ar a 25 °C e pressão atmosférica)
nu = mu / rho
A = np.sqrt(nu / U)
p = 0.001
h = 0

# Solução Numérica da Equação de Blasius usando o método de Runge-Kutta
f1 = lambda x, y1, y2, y3: y2
f2 = lambda x, y1, y2, y3: y3
f3 = lambda x, y1, y2, y3: -y1 * y3
eta = np.arange(0, 10 + p, p)
x = np.arange(0, 10 + p, p)
h1 = np.zeros_like(eta)
h2 = np.zeros_like(eta)
h3 = np.zeros_like(eta)
h3[0] = 0.4696

for i in range(len(eta) - 1):
  #Calcula a inclinação inicial da curva:
  k1 = p * np.array([f1(eta[i], h1[i], h2[i], h3[i]), f2(eta[i], h1[i], h2[i], h3[i]), f3(eta[i], h1[i],h2[i], h3[i])])
  #Calcula a inclinação nos pontos intermediários:
  k2 = h * np.array([f1(eta[i], h1[i] + k1[0] / 2, h2[i] + k1[1] / 2, h3[i] + k1[2] / 2), f2(eta[i] + p / 2, h1[i] + k1[0] / 2, h2[i] + k1[1] / 2, h3[i] + k1[2] / 2), f3(eta[i] + p / 2, h1[i] + k1[0] / 2, h2[i] + k1[1] / 2, h3[i] + k1[2] / 2)])
  k3 = h * np.array([f1(eta[i], h1[i] + k2[0] / 2, h2[i] + k2[1] / 2, h3[i] + k2[2] / 2), f2(eta[i] + p / 2, h1[i] + k2[0] / 2, h2[i] + k2[1] / 2, h3[i] + k2[2] / 2), f3(eta[i] + p / 2, h1[i] + k2[0] / 2, h2[i] + k2[1] / 2, h3[i] + k2[2] / 2)])
  #Calcula a inclinação final:
  k4 = p * np.array([f1(eta[i], h1[i] + k3[0], h2[i] + k3[1], h3[i] + k3[2]), f2(eta[i] + p, h1[i] + k3[0], h2[i] + k3[1], h3[i] + k3[2]), f3(eta[i] + p, h1[i] + k3[0], h2[i] + k3[1], h3[i] + k3[2])])
  #Calculo da média ponderada:
  h3[i + 1] = h3[i] + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
  h2[i + 1] = h2[i] + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
  h1[i + 1] = h1[i] + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6

# Plotagem dos Gráficos
plt.figure(1)
plt.plot(h1, eta, label='f(η)', linewidth=2)
plt.plot(h2, eta, label="f'(η)", linewidth=2)
plt.plot(h3, eta, label="f''(η)", linewidth=2)
plt.xlim([0, 2])
plt.title('Solução da Equação de Blasius', fontsize=14)
plt.xlabel('f, f\' e f\'\'', fontsize=20)
plt.ylabel('η', fontsize=20)
plt.grid(True)
plt.legend(fontsize=14)
plt.xlim(-0.05, 2)
#plt.ylim(-0.05, 6)
# Perfil de Velocidade e Distribuição da Espessura da Camada Limite
plt.figure(2)
delta = 5 * np.sqrt(x) * A
plt.plot(x, delta, label='δ(x)', color='black', linewidth=2)
pontos = [2, 4, 8]
for pontos in pontos:
  y = eta * np.sqrt(pontos) * A
  plt.plot(h2 + pontos, y, linewidth=2)
plt.title('Perfil de Velocidade em x = 2, 4 e 8, com camada limite', fontsize=14)
plt.xlabel('x (m)', fontsize=20)
plt.ylabel('y (m)', fontsize=20)
plt.legend(['δ(x)', 'x = 2', 'x = 4', 'x = 8'], fontsize=14)
#plt.ylim([0, 2 * np.max(y)])
plt.grid(True)
#plt.xlim(-0.05, 2)
plt.ylim(0, 0.1)
# Exibição dos Gráficos
plt.show()
