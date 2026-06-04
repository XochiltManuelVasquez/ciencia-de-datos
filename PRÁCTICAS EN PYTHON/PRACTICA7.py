# Práctica 7: Implementación de Modelos de Machine Learning

# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Crear datos
np.random.seed(789)

n = 500

df = pd.DataFrame({
    'experiencia': np.random.normal(5, 2, n),
    'certificaciones': np.random.poisson(3, n),
    'habilidades': np.random.uniform(1, 10, n),
    'salario': np.random.normal(50000, 10000, n)
})

# K-Means
scaler = StandardScaler()
X = scaler.fit_transform(df[['experiencia', 'salario']])

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Gráfica única
plt.scatter(df['experiencia'], df['salario'], c=df['cluster'])

plt.title('Clusters de empleados')
plt.xlabel('Experiencia')
plt.ylabel('Salario')

plt.show()