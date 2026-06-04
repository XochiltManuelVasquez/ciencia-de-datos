# Práctica 2: Análisis de Calidad Industrial 

# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Crear datos
np.random.seed(42)

n = 500

temperatura = np.random.normal(75, 5, n)
tasa_error = (temperatura * 0.3) + np.random.normal(0, 2, n)
turno = np.random.choice(['Matutino', 'Vespertino'], n)

df = pd.DataFrame({
    'temperatura': temperatura,
    'tasa_error': tasa_error,
    'turno': turno
})

# Gráfica 1
plt.boxplot([
    df[df['turno'] == 'Matutino']['tasa_error'],
    df[df['turno'] == 'Vespertino']['tasa_error']
])

plt.xticks([1, 2], ['Matutino', 'Vespertino'])
plt.title('Errores por Turno')
plt.ylabel('Tasa de Error')
plt.show()

# Gráfica 2
plt.scatter(df['temperatura'], df['tasa_error'])

plt.title('Temperatura vs Error')
plt.xlabel('Temperatura')
plt.ylabel('Tasa de Error')
plt.show()