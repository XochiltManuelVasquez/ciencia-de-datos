# Práctica 8: Clasificación de Intrusiones con KNN y Logística

# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Crear datos
np.random.seed(444)

n = 400

df = pd.DataFrame({
    'latencia': np.random.normal(20, 5, n),
    'fallos': np.random.poisson(1, n),
    'paquete': np.random.normal(500, 100, n)
})

# K-Means
scaler = StandardScaler()
X = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Gráfica única
plt.scatter(df['latencia'], df['fallos'], c=df['cluster'])

plt.title('Clusters de red')
plt.xlabel('Latencia')
plt.ylabel('Fallos')

plt.show()
