import numpy as np
results_path = '/tmp/'
np.savez(os.path.join(results_path, "run_%s_%s.npz" % ("wasserstein" if args.wasserstein else "regular",
                                                       "ml" if args.numz == 1 else "bayes")),
         **results)