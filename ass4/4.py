import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def density (x):
    return np.exp (-x**2/2) * (np.sin (x)**2 + 3 * np.cos (x)**2 * np.sin (7*x)**2 + 1)

def rejection_sampling (density, sample_distribution, k, n, sampler_args=(-3,3)):
    samples = []
    while len (samples) < n:
        z = sample_distribution (*sampler_args)
        p_z = density (z)
        u = np.random.uniform (0,1)
        if u <= p_z/(k):
            samples.append (z)
    return samples

    

if __name__ == "__main__":
    x = np.linspace (-3,3,100000)
    y = density (x)
    #y = y / np.max (y)
    plt.plot (x,y/np.max(y), label="normalised density")
    #plt.show()
    
    uniform_k = np.max (y) * 6
    print ("uniform k: ", uniform_k)
    uniform_samples = rejection_sampling (density, np.random.uniform, uniform_k, 1000)
    plt.hist (uniform_samples, bins=200, density=True, label="uniform samples")

    
    normal_k = 4 / stats.norm(0,1).pdf (0)
    print ("normal k: ", normal_k)
    normal_plot = stats.norm (0,1).pdf (x) * normal_k
    plt.plot (x, normal_plot / np.max(normal_plot), label="normal pdf")
    normal_samples = rejection_sampling (density, np.random.normal, normal_k, 1000, (0,1))
    
    plt.hist (normal_samples, bins=200, density=True, label="normal samples")
    plt.legend()
    plt.show()
