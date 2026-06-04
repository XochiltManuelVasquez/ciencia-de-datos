# Práctica 5: Reducción de Dimensiones en Tráfico de Red (Cyber-Systems)

# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Crear datos
np.random.seed(123)

n = 300

df = pd.DataFrame({
    'duracion': np.random.normal(50, 10, n),
    'paquetes': np.random.normal(100, 20, n),
    'errores': np.random.poisson(2, n),
    'latencia': np.random.normal(15, 5, n),
    'jitter': np.random.normal(2, 0.5, n),
    'cpu': np.random.normal(40, 10, n),
    'memoria': np.random.normal(40, 10, n),
    'http': np.random.normal(200, 50, n)
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
plt.title('PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()