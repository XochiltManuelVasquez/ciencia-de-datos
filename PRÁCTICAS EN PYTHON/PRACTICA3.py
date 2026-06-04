# Práctica 3: Análisis de Eficiencia en Logística Global

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Crear datos
np.random.seed(42)

n = 600

distancia = np.random.normal(500, 120, n)
tiempo = distancia * 0.08 + np.random.normal(0, 3, n)
transporte = np.random.choice(['Terrestre', 'Aéreo'], n)

df = pd.DataFrame({
    'distancia': distancia,
    'tiempo': tiempo,
    'transporte': transporte
})

# Gráfica 1
for tipo in df['transporte'].unique():
    datos = df[df['transporte'] == tipo]
    plt.scatter(datos['distancia'], datos['tiempo'], label=tipo)

plt.title('Distancia y Tiempo')
plt.xlabel('Distancia')
plt.ylabel('Tiempo')
plt.legend()
plt.show()

# Gráfica 2
plt.boxplot([
    df[df['transporte'] == 'Terrestre']['tiempo'],
    df[df['transporte'] == 'Aéreo']['tiempo']
])

plt.xticks([1, 2], ['Terrestre', 'Aéreo'])
plt.title('Tiempo por Transporte')
plt.ylabel('Tiempo de Entrega')
plt.show()