import matplotlib.pyplot as plt
def plot_psd_example(psds,ax=None,system_name='system_0'):
    if ax is None:
        fig, ax = plt.subplots()
    for psd in psds:
        ax.plot(psd, alpha=0.1,color='blue')
    ax.set_title('PSD of healthy '+ system_name)
    ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    return ax