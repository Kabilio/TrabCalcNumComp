import numpy as np
from scipy.integrate import quad
from scipy.stats import linregress
import matplotlib.pyplot as plt

# Dados coletados: tempo (segundos) e velocidade (m/s)
tempo_medido = np.array([0, 2, 4, 6, 8, 10])  # em segundos
velocidade_medida = np.array([0, 4, 8, 12, 16, 20])  # em m/s

# Interpolação polinomial
coeficientes = np.polyfit(tempo_medido, velocidade_medida, 3)  # polinômio de grau 3

# Função para calcular a velocidade em qualquer tempo
def velocidade_interpolada(tempo):
    return np.polyval(coeficientes, tempo)

def calcular_velocidade_para_tempo_nao_medido():
    while True:
        try:
            tempo_nao_medido = float(
                input("Digite o tempo não medido (em segundos): "))
            if tempo_nao_medido < 0:
                print("Tempo não pode ser negativo. Tente novamente.")
                continue
            velocidade_estimada = velocidade_interpolada(tempo_nao_medido)
            print(f"A velocidade estimada em {tempo_nao_medido}s é {velocidade_estimada:.2f} m/s")
        except ValueError:
            print("Erro: tempo deve ser um número. Tente novamente.")
        except KeyboardInterrupt:
            print("\nOperação cancelada pelo usuário.")
            break

# Teste da função
calcular_velocidade_para_tempo_nao_medido()

# Função para calcular a distância percorrida durante o teste de 0 a t segundos
def calcular_distancia_percurso(t):
    # Função de velocidade
    v = lambda x: velocidade_interpolada(x)

    # Cálculo da distância usando integração numérica
    distancia, erro = quad(v, 0, t)  # integração para calcular a distância
    return distancia

# Calcular a distância total percorrida durante os 10 segundos de teste
distancia_total = calcular_distancia_percurso(10)
print(f"A distância total percorrida durante os 10 segundos de teste é {distancia_total:.2f} metros.")

# Ajuste de uma função linear (reta) à relação tempo-velocidade
coeficientes_linear = np.polyfit(tempo_medido, velocidade_medida, 1)  # polinômio de grau 1 (função linear)

# Coeficientes da função linear
a, b = coeficientes_linear

# Velocidade predita pela função linear
velocidade_predita = np.polyval(coeficientes_linear, tempo_medido)

print("Plotando dados...")
# Plotagem dos dados e da função linear ajustada
plt.figure(figsize=(10, 6))
plt.plot(tempo_medido, velocidade_medida, 'bo', label='Dados Medidos')
plt.plot(tempo_medido, velocidade_predita, 'r-', label='Função Linear Ajustada')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (m/s)')
plt.title('Ajuste de uma Função Linear à Relação Tempo-Velocidade')
plt.legend()
plt.grid(True)
plt.show()

# Calcular a aceleração média
aceleracao_media = (velocidade_medida[-1] - velocidade_medida[0]) / (
    tempo_medido[-1] - tempo_medido[0])

# Verificar se a aceleração é constante
aceleracoes = np.diff(velocidade_medida) / np.diff(tempo_medido)
aceleracao_media = np.mean(aceleracoes)
aceleracao_constante = np.allclose(aceleracoes, aceleracao_media)
if aceleracao_constante:
    print("A aceleração do carro é constante.")
else:
    print("A aceleração do carro não é constante.")

# Função da velocidade em relação ao tempo
def velocidade_funcao(t):
    return np.polyval(coeficientes, t)

# Metade da velocidade máxima
metade_velocidade_maxima = max(velocidade_medida) / 2

# Tempo em que o carro alcança a metade da velocidade máxima
def encontrar_tempo_metade_velocidade(max_velocidade):
    def funcao_tempo(t):
        return velocidade_funcao(t) - max_velocidade
    from scipy.optimize import root_scalar
    resultado = root_scalar(funcao_tempo, bracket=[0, 10], method='brentq')
    return resultado.root

tempo_alcance_metade = encontrar_tempo_metade_velocidade(metade_velocidade_maxima)
print(f"O carro alcança a metade da velocidade máxima em {tempo_alcance_metade:.2f} segundos.")

# Adicionar análise estatística para avaliar a qualidade do ajuste linear
slope, intercept, r_value, p_value, std_err = linregress(tempo_medido, velocidade_medida)
