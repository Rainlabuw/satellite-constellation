import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    T = 100
    m = 3
    true_moisture_hist = np.zeros((m,T))

    true_moisture_base = np.random.uniform(3,10,m)
    freq = np.array([10,50,3])
    for k in range(T):
        true_moisture_hist[0,k] = true_moisture_base[0] + 2*np.sin(k/freq[0]*np.pi) + np.random.normal(0,0.1)
        true_moisture_hist[1,k] = true_moisture_base[1] + 1*np.sin(k/freq[1]*np.pi) + np.random.normal(0,0.1)
        true_moisture_hist[2,k] = true_moisture_base[2] + 0.5*np.sin(k/freq[2]*np.pi) + np.random.normal(0,0.1)

    var = np.ones(m)
    est = np.zeros(m)

    sensor_cov = 0.1

    var_hist = np.zeros((m,T))
    est_hist = np.zeros((m,T))

    var_add = 1/freq/10

    for k in range(T):
        max_var_idx = np.argmax(var)

        #get measurement of max var channel
        measurement = true_moisture_hist[max_var_idx,k] + np.random.normal(0,sensor_cov)

        #Perform bayesian update of estimate and variance based on new estimate
        new_cov = 1/(1/var[max_var_idx] + 1/sensor_cov)
        new_est = new_cov*(est[max_var_idx]/var[max_var_idx] + measurement/sensor_cov)

        var[max_var_idx] = new_cov
        est[max_var_idx] = new_est

        var += var_add

        var_hist[:,k] = var
        est_hist[:,k] = est

    fig, axes = plt.subplots(2,1)
    axes[0].plot(var_hist[0,:], 'b')
    axes[0].plot(var_hist[1,:], 'g')
    axes[0].plot(var_hist[2,:], 'r')
    axes[0].set_ylim([0,0.2])

    axes[1].plot(est_hist[0,:], 'b')
    axes[1].plot(est_hist[1,:], 'g')
    axes[1].plot(est_hist[2,:], 'r')
    axes[1].plot(true_moisture_hist[0,:], 'b--')
    axes[1].plot(true_moisture_hist[1,:], 'g--')
    axes[1].plot(true_moisture_hist[2,:], 'r--')

    plt.show()