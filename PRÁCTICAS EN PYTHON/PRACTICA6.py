# Práctica 6: Optimización de Sensores en Smart Cities

# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Crear datos
np.random.seed(456)

n = 400

df = pd.DataFrame({
    'temp': np.random.normal(28, 4, n),
    'humedad': np.random.normal(60, 10, n),
    'vehiculos': np.random.normal(120, 30, n),
    'viento': np.random.normal(15, 5, n),
    'solar': np.random.normal(800, 100, n),
    'peatones': np.random.normal(50, 15, n)
})

# PCA
scaler = StandardScaler()
X = scaler.fit_transform(df)

pca = PCA(n_components=2)
resultado = pca.fit_transform(X)

# Gráfica 1 (varianza)
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Varianza acumulada')
plt.xlabel('Componentes')
plt.ylabel('Varianza')
plt.grid()
plt.show()

# Gráfica 2 (PCA)
plt.scatter(resultado[:, 0], resultado[:, 1])
plt.title('PCA urbano')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()