# Reporte Técnico: Reducción de Dimensionalidad en Monitoreo Industrial

# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Crear datos
np.random.seed(42)

n = 200

df = pd.DataFrame({
    'temp1': np.random.normal(100, 5, n),
    'temp2': np.random.normal(100, 5, n),
    'presion': np.random.normal(50, 10, n),
    'vibracion': np.random.normal(20, 3, n),
    'ruido': np.random.normal(60, 5, n),
    'voltaje': np.random.normal(220, 2, n)
})

# PCA
scaler = StandardScaler()
X = scaler.fit_transform(df)

pca = PCA()
X_pca = pca.fit_transform(X)

# Gráfica 1 (varianza)
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.axhline(0.85, linestyle='--')

plt.title('Varianza acumulada')
plt.xlabel('Componentes')
plt.ylabel('Varianza')

plt.grid()
plt.show()

# Gráfica 2 (PCA)
plt.scatter(X_pca[:, 0], X_pca[:, 1])

plt.title('PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.show()