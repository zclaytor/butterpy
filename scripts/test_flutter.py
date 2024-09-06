import numpy as np
import butterpy as bp

if __name__ == "__main__":
    np.random.seed(88)
    f = bp.Flutter(n_sims=10, duration=4*365)
    f.sample()
    f.run()

