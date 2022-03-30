def serialize(x):
    """recursive function that converts nested arrays to lists and the numpy
    numeric data types to nativep ython floats to make the structure json
    serializable, so that it can be dumped to json. input is iterable output is
    python list"""
    out = []
    for k in x:
        try:
            if len(k) > 0:
                out.append(serialize(list(k)))
        except TypeError:
            out.append(float(k))
    return out

try:
    from gwtools import mismatch
    from gwtools import gwutils
    import numpy as np
except:
    print('gwtools not installed, cant calculate SNR')

def snr_whitenoise(signal,signal_with_noise,R,length=1):
    WaveformLength = length
    dt = WaveformLength/len(signal)
    freqs,hFreq = gwutils.freqDomainWaveform(signal,dt)
    freqs_noise,nFreq = gwutils.freqDomainWaveform(signal_with_noise,dt)
    df = 1/WaveformLength
    noise_std = R 
    Sn = 2.0 * dt* (noise_std ** 2.0) / WaveformLength
    numerator = np.real(mismatch.inner_complex(hFreq, nFreq, psd=Sn, df=df,ligo=True, pos_freq=False))
    denominator = np.sqrt(mismatch.inner(hFreq, hFreq, psd=Sn, df=df, ligo=True, pos_freq=False))
    match_filter_snr = numerator / denominator
    return match_filter_snr
 
