import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.interpolate import CubicSpline, PchipInterpolator
import pdb


def p_BT(y, maxlag, nfft, fs, lagwindow):
    x = y-np.mean(y)
    ryy = np.correlate(x, x, mode='full')[len(x)-maxlag-1:len(x)+maxlag]
    ryy /= len(x)
    ryy = ryy

    if lagwindow == 't':
        w = np.arange(1./(maxlag+1), 1, 1./(maxlag+1))
        w = np.concatenate([w, [1], w[::-1]])
    else:
        assert NotImplementedError
    Syy = np.abs(np.fft.fft(w*ryy, nfft))
    deltf = fs/nfft
    f= np.arange(0, fs-deltf, deltf)
    return Syy, f


def get_modulation_index(psth, psth_duration=1.5, TF=6):
    # interpolate input psth
    interpfactor = 2
    n = len(psth)
    v = psth
    x = np.arange(0, (n-1)*psth_duration/n*1.001, psth_duration/n)
    xq = np.arange(0, (n-1)*psth_duration/n*1.001, psth_duration/(interpfactor*n))
    cs = PchipInterpolator(x, v)
    psth = cs(xq)
    N = len(psth)

    ## compute F1z

    # demean signal
    ds = psth
    baseline = np.mean(ds)
    ds = ds-baseline

    N_f = 3*N
    T_s = psth_duration/N
    f_s = 1/T_s
    maxlag = int(np.fix(N/3))
    lagwindow = 't'
    [pow_v, f] = p_BT(ds, maxlag, N_f, f_s, lagwindow)

    # find stimulus frequency vector index
    fidx = np.where(f<=TF)[0]
    fidx = fidx[-1]
    spect = pow_v[fidx] 
    sigspect = np.std(pow_v) 
    meanspect = np.mean(pow_v)
    F1z = (spect-meanspect) / sigspect

    return F1z, None


def main():
    psth = [
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
    4.8857,
    8.0571,
    7.3714,
    1.9714,
    1.9714,
    7.1143,
    8.8286,
    7.1143,
    1.9714,
    1.9714,
    7.1143,
    8.8286,
    7.1143,
    1.9714,
    1.9714,
    7.1143,
    8.8286,
    7.1143,
    1.9714,
    1.9714,
    7.1143,
    8.8286,
    7.1143,
    1.9714,
    1.9714,
    7.1143,
    8.8286,
    7.1143,
    1.9714,
    1.9714,
    7.1143,
    8.8286,
    7.1143,
    1.9714,
    1.9714,
    7.1143,
    7.4571,
    6.0857,
    ]
    print(get_modulation_index(psth))


if __name__ == '__main__':
    main()
