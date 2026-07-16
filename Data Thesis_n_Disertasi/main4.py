import numpy as np

x = np.load("polynomial_pq_hybrid_output/p4_q6/branch_profiles.npz")

print(x.files)

for k in x.files:
    print(k, x[k].shape, x[k].dtype)
