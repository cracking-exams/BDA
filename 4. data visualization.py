# ============================================
# 📊 Iris Dataset — Complete Visualization Suite
# Covers: a–g (1D to Network Visualization)
# ============================================

# Imports
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import squarify

# Load Dataset
iris = sns.load_dataset('iris')

# --------------------------------------------
# a. 1D (Linear) Data Visualization
# --------------------------------------------
plt.figure(figsize=(6,4))
sns.histplot(iris['sepal_length'], kde=True, color='skyblue', bins=15)
plt.title('1D Visualization: Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Count')
plt.show()

# --------------------------------------------
# b. 2D (Planar) Data Visualization
# --------------------------------------------
plt.figure(figsize=(6,4))
sns.scatterplot(x='sepal_width', y='sepal_length', hue='species', data=iris)
plt.title('2D Visualization: Sepal Length vs Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Sepal Length (cm)')
plt.show()

# --------------------------------------------
# c. 3D (Volumetric) Data Visualization
# --------------------------------------------
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

for species, color in zip(iris['species'].unique(), ['red', 'green', 'blue']):
    subset = iris[iris['species'] == species]
    ax.scatter(subset['sepal_length'], subset['sepal_width'], subset['petal_length'],
               label=species, alpha=0.7, s=50, color=color)

ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.set_zlabel('Petal Length (cm)')
plt.title('3D Visualization: Sepal & Petal Dimensions')
plt.legend()
plt.show()

# --------------------------------------------
# d. Temporal Data Visualization (Simulated)
# --------------------------------------------
iris_sorted = iris.sort_values(by='sepal_length').reset_index(drop=True)
iris_sorted['index'] = iris_sorted.index + 1

plt.figure(figsize=(7,4))
sns.lineplot(x='index', y='sepal_length', hue='species', data=iris_sorted)
plt.title('Temporal Visualization: Sepal Length Across Samples (Simulated Time)')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.show()

# --------------------------------------------
# e. Multidimensional Data Visualization
# --------------------------------------------
sns.pairplot(iris, hue='species', diag_kind='kde')
plt.suptitle('Multidimensional Visualization: Pairwise Feature Relationships', y=1.02)
plt.show()

# --------------------------------------------
# f. Tree / Hierarchical Data Visualization
# --------------------------------------------
grouped = iris.groupby('species')[['sepal_length']].mean().reset_index()

plt.figure(figsize=(6,4))
squarify.plot(sizes=grouped['sepal_length'], label=grouped['species'], alpha=0.7)
plt.title('Tree / Hierarchical Visualization: Mean Sepal Length by Species')
plt.axis('off')
plt.show()

# --------------------------------------------
# g. Network Data Visualization (Simulated)
# --------------------------------------------
# Compute similarity (inverse of distance) between species based on mean features
means = iris.groupby('species').mean()
dist = pd.DataFrame(np.linalg.norm(means.values[:, None] - means.values, axis=2),
                    index=means.index, columns=means.index)

# Build network graph
G = nx.Graph()
for i in dist.index:
    for j in dist.columns:
        if i != j:
            G.add_edge(i, j, weight=round(1/(dist.loc[i, j]+1), 2))

# Draw network
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(6,5))
nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=2000, edge_color='gray', font_size=10)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Network Visualization: Species Similarity Graph')
plt.show()

# ============================================
# ✅ End of Visualizations
# ============================================
