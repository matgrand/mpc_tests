import numpy as np; π = np.pi
import matplotlib.pyplot as plt

def linear_resample(iu, t): 
    '''
    input iu is an approxximation of the control input
    Expand the control input to match the state vector
    iu: compressed input, t: simulation time, ne: number expanded control inputs
    '''
    nc, ne = len(iu), len(t) # number of compressed control inputs
    assert nc <= ne, f'input must be smaller than ne nc: {nc}, ne: {ne}'
    ou = np.zeros((ne)) # expanded control input
    ct = np.linspace(0, t[-1], nc) # compressed time
    et = np.linspace(0, t[-1], ne) # expanded time
    ii = 0 # index for the compressed input
    for i in range(ne):
        if et[i] > ct[ii+1]: ii += 1 # update the index
        ia, ib = ct[ii], ct[ii+1] # time interval for the compressed input
        a, b = iu[ii], iu[ii+1] # control input interval
        ou[i] = a + (et[i] - ia)*(b - a)/(ib - ia) # linear interpolation
    return ou

def addittive_resample(iu, t): 
    '''
    input is defined as a sequence of additions to the first control input
    Expand the control input to match the state vector
    iu: input, t: simulation time, ne: number expanded control inputs
    '''
    nc, ne = len(iu), len(t) # number of compressed control inputs
    assert nc <= ne, f'input must be smaller than ne nc: {nc}, ne: {ne}'
    ou = np.zeros((ne)) # expanded control input
    ct = np.linspace(0, t[-1], nc) # input time
    et = np.linspace(0, t[-1], ne) # expanded time
    ii = 0 # index for the compressed input
    cumulated = iu[0] # cumulated control input
    for i in range(ne):
        if et[i] > ct[ii+1]: 
            ii += 1 # update the index
            cumulated += iu[ii] # update the cumulated control input
        dtc = ct[ii+1] - ct[ii] # time interval for the compressed input
        dti = et[i] - ct[ii] # time interval for the expanded input
        ou[i] = cumulated + iu[ii+1]*dti/dtc # linear interpolation
    return ou

# def frequency_resample(iu, t, max_freq=8): 
#     '''
#     input is defined as a sequence of additions to the first control input
#     Expand the control input to match the state vector
#     iu: input, t: simulation time, ne: number expanded control inputs
#     '''
#     nc, ne = len(iu), len(t) # number of compressed control inputs
#     et = np.linspace(0, t[-1], ne) # expanded time
#     freqs = -1 + np.logspace(0, np.log10(max_freq), nc-1) # frequencies
#     signal = np.sum([iu[i+1]*np.sin(2*π*freqs[i]*et+0.2*iu[0]) for i in range(nc-1)], axis=0) # add the sinusoids
#     return signal # add the first control input as an offset

# def frequency_resample(iu, t, max_freq=8): 
#     '''
#     input is defined as a sequence of additions to the first control input
#     Expand the control input to match the state vector
#     iu: input, t: simulation time, ne: number expanded control inputs
#     '''
#     nc, ne = len(iu), len(t) # number of compressed control inputs
#     et = np.linspace(0, t[-1], ne) # expanded time
#     freqs = -1 + np.logspace(0, np.log10(max_freq), nc) # frequencies
#     return np.sum([iu[i]*np.sin(2*π*freqs[i-1]*et) for i in range(nc)], axis=0) # add the sinusoids

def frequency_resample(iu, t, max_freq=8): 
    '''
    input is defined as a sequence of additions to the first control input
    Expand the control input to match the state vector
    iu: input, t: simulation time, ne: number expanded control inputs
    '''
    nc, ne = len(iu), len(t) # number of compressed control inputs
    et = np.linspace(0, t[-1], ne) # expanded time
    freqs = -1 + np.logspace(0, np.log10(max_freq), nc) # frequencies
    s = np.sum([iu[i]*np.cos(2*π*freqs[i-1]*et) for i in range(0, nc, 2)], axis=0) # add the sinusoids
    c = np.sum([iu[i]*np.sin(2*π*freqs[i-1]*et) for i in range(1, nc, 2)], axis=0) # add the cosinoids
    return s + c


if __name__  == '__main__':
    # test the resampling
    ni = 17
    ne = 100

    ti = np.linspace(0, 1, ni)
    te = np.linspace(0, 1, ne)
    iu = 0.1 + np.sin(3*π*ti) # compressed input

    lreu = linear_resample(iu, te) # linear resample
    adreu = addittive_resample(iu, te) # addittive resample
    frreu = frequency_resample(iu, te) # frequency resample

    istyle = '--o'
    ostyle = '-'
    ci = 'b'
    co = 'r'

    fig, ax = plt.subplots(3,1, figsize=(12,12))
    ax[0].plot(ti, iu, istyle, label='compressed input', color=ci)
    ax[0].plot(te, lreu, ostyle, label='expanded input', color=co)
    for i in range(ni): ax[0].axvline(ti[i], lw=0.5, color=ci)
    for i in range(ne): ax[0].axvline(te[i], lw=0.5, color=co)
    ax[0].set_title(f'linear resample: ni: {ni}, ne: {ne}')
    ax[1].plot(ti, iu, istyle, label='compressed input', color=ci)
    ax[1].plot(te, adreu, ostyle, label='expanded input', color=co)
    for i in range(ni): ax[1].axvline(ti[i], lw=0.5, color=ci)
    for i in range(ne): ax[1].axvline(te[i], lw=0.5, color=co)
    ax[1].set_title(f'addittive resample: ni: {ni}, ne: {ne}')
    ax[2].plot(ti, iu, istyle, label='compressed input', color=ci)
    ax[2].plot(te, frreu, ostyle, label='expanded input', color=co)
    for i in range(ni): ax[2].axvline(ti[i], lw=0.5, color=ci)
    for i in range(ne): ax[2].axvline(te[i], lw=0.5, color=co)
    ax[2].set_title(f'frequency resample: ni: {ni}, ne: {ne}')

    plt.show()
