# Práctica 4: Reducción de Dimensionalidad en Monitoreo Industrial

# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Crear datos
np.random.seed(42)

n = 500

df = pd.DataFrame({
    'temp': np.random.normal(80, 5, n),
    'presion': np.random.normal(30, 3, n),
    'vibracion': np.random.normal(10, 2, n),
    'gas': np.random.normal(50, 8, n)
})

# PCA
scaler = StandardScaler()
datos = scaler.fit_transform(df)

pca = PCA(n_components=2)
resultado = pca.fit_transform(datos)

# Gráfica 1
plt.plot(
    np.cumsum(pca.explained_variance_ratio_),
    marker='o'
)

plt.title('Varianza PCA')
plt.xlabel('Componentes')
plt.ylabel('Varianza')
plt.grid()
plt.show()

# Gráfica 2
plt.scatter(resultado[:, 0], resultado[:, 1])

plt.title('PCA')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.show()