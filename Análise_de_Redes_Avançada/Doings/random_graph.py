import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

N = 1000
p = 0.1

g = nx.erdos_renyi_graph(N, p)
k_medio_e = np.mean(list(dict(g.degree()).values()))
k_medio_t = p * (N - 1)
sigma_e = np.std(list(dict(g.degree()).values()))
sigma_t = np.sqrt(k_medio_t)
x_min = int(k_medio_e - 3 * sigma_e)
x_max = int(k_medio_e + 3 * sigma_e)

degree_sequence = list(dict(g.degree()).values())
d = np.bincount(degree_sequence)
x = np.arange(len(d))
plt.bar(x, d, width=1.0, color='b', alpha=0.7)
poisson_vals = poisson.pmf(x, k_medio_e)
plt.plot(x, poisson_vals * N, 'ro-', linewidth=2, label='Poisson Distribution', color='red')
plt.xlim(x_min, x_max)
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.legend()
plt.show()

RMSE = np.sqrt(np.mean((d - poisson.pmf(x, k_medio_e) * N) ** 2))
c_i = nx.transitivity(g)
print("Mean Clustering Coefficient:", c_i)
print("Mean Distance:", nx.average_shortest_path_length(g))
print("Estimated Power Law Exponent:", np.log(N) / np.log(k_medio_e))