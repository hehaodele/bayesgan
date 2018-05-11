import numpy as np
from os.path import join
from matplotlib.pyplot import *

x_dim = 100
z_dim = 10
numz = 10

results_path = '/tmp/synth_results'
ml_path = join(results_path, 'experiment_1521835967', 'run_regular_ml.npz')
bayes_path = join(results_path, 'experiment_1521851980', 'run_regular_bayes.npz')

results = np.load(join(ml_path))
bresults = np.load(join(bayes_path))

figure()
plot(np.arange(100, 1001, 100), results["divergences"], '--k', linewidth=2, label="ML GAN")
plot(np.arange(100, 1001, 100), bresults["divergences"], '-g', linewidth=2, label="Bayes GAN")
ylim([0.0, 0.6])
title("JS div. (D = %i)" % x_dim)
legend(loc='upper right')
xlabel("No. of iterations")
ylabel("JS Divergence")
show()