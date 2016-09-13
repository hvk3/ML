KMeans runs for each dataset for atmost 20 iterations, even though the cost function converges after atmost 5-6 iterations.
Required dependencies : matplotlib, numpy, sklearn, copy, random and re

To run, enter appropriate dataset and K after running python main.py. The script will run both GMM and K-means on the required dataset, with the given number of clusters K.
Uncommnent lines 109-111 to get the cost vs iteration plots.