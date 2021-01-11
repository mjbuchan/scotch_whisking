def load_data(date, path):

    '''
    need to comment up
    '''

    import os
    from scipy.io import loadmat
    import numpy as np
    import scipy.ndimage as nd

    # load waveform data

    unit_waveforms = loadmat(os.path.join(path, date, 'unit_waveforms.mat'))
    unit_waveforms = unit_waveforms['unit_waveforms'] 

    # load lfp data

    lfp_1 = loadmat(os.path.join(path, date, 'whisk_dual_single/lfp_1.mat'))
    lfp_1 = lfp_1['lfp_1']

    lfp_2 = loadmat(os.path.join(path, date, 'whisk_dual_single/lfp_2.mat'))
    lfp_2 = lfp_2['lfp_2']

    # load spikes from 10ms LED flash protocol

    opto_tag_10 = loadmat(os.path.join(path, date, 'opto_tag_10/spikes.mat'))
    opto_tag_10 = opto_tag_10['spikes']

    # load depths of sorted units

    unit_depths = loadmat(os.path.join(path,date, 'unit_depths.mat'))
    unit_depths = unit_depths['unit_depths'].flatten()

    # load spontaneous data

    spont_spikes = loadmat(os.path.join(path, date, 'spontaneous/spikes.mat'))
    spont_lfp = loadmat(os.path.join(path,date, 'spontaneous/lfp.mat'))
    spont_mua_spikes = loadmat(os.path.join(path, date, 'spontaneous/mua_spikes.mat'))

    spont_lfp = np.mean(spont_lfp['lfp'], 0)
    spont_spikes = spont_spikes['spikes']
    spont_mua_spikes = spont_mua_spikes['mua_spikes']

    for i in range(len(spont_lfp)):

        spont_lfp[i] = nd.gaussian_filter1d(spont_lfp[i], sigma = 20)

    # load single whisk data

    single_whisk_1 = loadmat(os.path.join(path,date,'whisk_dual_single/spikes_1.mat'))
    single_whisk_2 = loadmat(os.path.join(path,date, 'whisk_dual_single/spikes_2.mat'))

    single_whisk_1 = single_whisk_1['spikes_1']
    single_whisk_2 = single_whisk_2['spikes_2']

    # load quad whisk data

    quad_whisk_1 = loadmat(os.path.join(path, date, 'whisk_dual_quad/spikes_1.mat'))
    quad_whisk_2 = loadmat(os.path.join(path, date, 'whisk_dual_quad/spikes_2.mat'))

    quad_whisk_1 = quad_whisk_1['spikes_1']
    quad_whisk_2 = quad_whisk_2['spikes_2']

    print('data loaded successfully - waveform, lfp, opto, depths, spont, single whisk, quad whisk')

    return unit_waveforms, lfp_1, lfp_2, opto_tag_10, unit_depths, spont_spikes, spont_lfp, spont_mua_spikes, single_whisk_1, single_whisk_2, quad_whisk_1, quad_whisk_2

def load_freq_data(path, date):
    
    import os
    from scipy.io import loadmat
    import numpy as np
    
    spikes_1_1 = loadmat(os.path.join(path, date, 'frequency/spikes_1_1'))
    spikes_1_1 = spikes_1_1['spikes_1_1']
    
    spikes_1_2 = loadmat(os.path.join(path, date, 'frequency/spikes_1_2'))
    spikes_1_2 = spikes_1_2['spikes_1_2']
                         
    spikes_1_3 = loadmat(os.path.join(path, date, 'frequency/spikes_1_3'))
    spikes_1_3 = spikes_1_3['spikes_1_3']
    
    spikes_1_4 = loadmat(os.path.join(path, date, 'frequency/spikes_1_4'))
    spikes_1_4 = spikes_1_4['spikes_1_4']
                         
    spikes_1_5 = loadmat(os.path.join(path, date, 'frequency/spikes_1_5'))
    spikes_1_5 = spikes_1_5['spikes_1_5']
                         
    freq_spikes_1 = [spikes_1_1, spikes_1_2, spikes_1_3, spikes_1_4, spikes_1_5]
                         


    spikes_2_1 = loadmat(os.path.join(path, date, 'frequency/spikes_2_1'))
    spikes_2_1 = spikes_2_1['spikes_2_1']
    
    spikes_2_2 = loadmat(os.path.join(path, date, 'frequency/spikes_2_2'))
    spikes_2_2 = spikes_2_2['spikes_2_2']
                         
    spikes_2_3 = loadmat(os.path.join(path, date, 'frequency/spikes_2_3'))
    spikes_2_3 = spikes_2_3['spikes_2_3']
    
    spikes_2_4 = loadmat(os.path.join(path, date, 'frequency/spikes_2_4'))
    spikes_2_4 = spikes_2_4['spikes_2_4']
                         
    spikes_2_5 = loadmat(os.path.join(path, date, 'frequency/spikes_2_5'))
    spikes_2_5 = spikes_2_5['spikes_2_5']
                         
    freq_spikes_2 = [spikes_2_1, spikes_2_2, spikes_2_3, spikes_2_4, spikes_2_5]
                         
    return freq_spikes_1, freq_spikes_2


def waveform_analysis(unit_waveforms):

    '''
    Matt Buchan // Akerman Lab - Sept 2020
    '''

    import numpy as np

    time = (np.arange(len(unit_waveforms[0,31,:]))/30000)*1000

    pop_t2p = []
    pop_trough_val = []
    pop_peak_val = []
    pop_half_width = []

    for i in range(len(unit_waveforms)):

        waveform_idx = np.argmin(np.min(unit_waveforms[i],1))

        der_waveform = (unit_waveforms[i,waveform_idx,:])

        trough = time[np.argmin(der_waveform[50:70])+50]
        peak = time[np.argmax(der_waveform[65:])+65]

        trough_value = np.min(der_waveform[50:70]+50)
        peak_value = np.max(der_waveform[65:]+65)

        d_t2p = peak-trough

        half_width_value = peak_value - (peak_value - trough_value)/2
        half_width_idx = np.argwhere(der_waveform < half_width_value) 
        time_1 = time[half_width_idx[0]]
        time_2 = time[half_width_idx[-1]]
        half_width = time_2 - time_1

        pop_t2p.append(d_t2p)
        pop_trough_val.append(trough_value)
        pop_peak_val.append(peak_value)
        pop_half_width.append(half_width)

    pop_t2p = np.array(pop_t2p)
    pop_trough_val = np.array(pop_trough_val)
    pop_peak_val = np.array(pop_peak_val)
    pop_half_width = np.array(pop_half_width)
    
    return pop_t2p, pop_trough_val, pop_peak_val, pop_half_width


def cluster_units(pop_t2p, pop_half_width):

    '''
    Matt Buchan // Akerman Lab - Sept 2020
    '''

    half_width_crit = (pop_half_width < 1)

    fs_units = ((pop_t2p > 0) & (pop_t2p < 0.5) & (half_width_crit[:,0] == True))
    rs_units = ((pop_t2p > 0.5) & (pop_t2p < 1.5)& (half_width_crit[:,0] == True))
    
    return fs_units, rs_units


def plot_unit_clusters(pop_t2p, pop_half_width, fs_units, rs_units):

    '''
    Matt Buchan // Akerman Lab - Sept 2020
    '''

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize = (3,3))

    sns.kdeplot(pop_t2p[fs_units], pop_half_width[fs_units], cmap = 'Greys', shade = True, shade_lowest = False)
    sns.kdeplot(pop_t2p[rs_units], pop_half_width[rs_units], cmap = 'Blues', shade = True, shade_lowest = False)

    plt.xlim(0, 1.5)
    plt.ylim(0,1)

    plt.yticks([0, 0.5, 1])

    plt.xlabel('Trough to peak (ms)')
    plt.ylabel('Half amplitude (ms)')

    #plt.vlines(0.5, 0, 1, ls = ':', color = 'grey')

    red = sns.color_palette("Greys")[-2]
    blue = sns.color_palette("Blues")[-2]
    plt.text(0.025, 0.5, "Fast-spiking", size=12, color=red)
    plt.text(0.125, 0.4, "n = {}".format(fs_units.sum()), size=12, color=red)
    plt.text(0.6, 0.9, "Regular-spiking", size=12, color=blue)
    plt.text(0.8, 0.8, "n = {}".format(rs_units.sum()), size=12, color=blue)

    print('fs units:', fs_units.sum())
    print('rs units:', rs_units.sum())

    #plt.tight_layout()



def calculate_csd(lfp_1, lfp_2, sigma, pw_ID):
    
    import scipy.ndimage as nd
    import numpy as np
    
    if pw_ID == 1: 

        lfp = lfp_1

        print('pw ID = 1, using lfp 1')

    if pw_ID == 2:

        lfp = lfp_2

        print('pw ID = 2, using lfp 2')

    csd = np.gradient(np.mean(lfp,1))
               
    csd = nd.gaussian_filter(csd[0], sigma)
    
    return csd  


def plot_csd(csd, window, date):

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize = (2,6))

    cbar_kws = {"orientation":"horizontal", 'aspect': 5, 'ticks': []
               }
    sns.heatmap(csd[:,1005:1005+window], cmap = 'jet', vmin=-100, vmax=100, cbar_kws = cbar_kws)

    plt.yticks([])
    plt.xticks([])

    plt.title(date)
    
    plt.tight_layout()



def calculate_l4(csd, unit_depths, date, method):

    import numpy as np

    avg = np.mean(csd[:, 1010:1011],1)[:22]
    
    sink_channels = [i for i,v in enumerate(avg) if v > 0]

    sink_max = np.argmax(np.mean(csd[:, 1010:1011],1)[:22])
    
    if method == 'auto_max':

        l4_top = 800 - (sink_max-4)*25
        l4_bottom = 800 - (sink_max+4)*25

    if method == 'auto_bound':

        if sink_channels[0] == 0:
            l4_top = 800-(sink_channels[4]*25)
        
        if sink_channels[0] != 0:
        
            l4_top = 800-(sink_channels[3]*25)
        
        l4_bottom = 800-(sink_channels[-3]*25)

        #l4_top = 800-(sink_max*25)+75
        #l4_bottom = 800-(sink_max*25)-75

    if method == 'manual':

        if date == '2020_06_23_1':

            l4_top = 800-(6*25)
            l4_bottom = 800-(18*25)

        if date == '2020_06_23_2':

            l4_top = 800-(8*25)
            l4_bottom = 800-(16*25)

        if date == '2020_06_24_1':

            l4_top = 800-(8*25)
            l4_bottom = 800-(16*25)

        if date == '2020_06_24_2':

            l4_top = 800-(8*25)
            l4_bottom = 800-(16*25)

        if date == '2020_06_26':

            l4_top = 800-(6*25)
            l4_bottom = 800-(18*25)

        if date == '2020_06_27':

            l4_top = 800-(8*25)
            l4_bottom = 800-(16*25)

        if date == '2020_06_28':

            l4_top = 800-(6*25)
            l4_bottom = 800-(18*25)

        if date == '2020_06_29':

            l4_top = 800-(8*25)
            l4_bottom = 800-(16*25)

    if method == 'spec':

        l4_top = 600
        l4_bottom = 400

    l23_bottom = l4_top

    l23_top = l4_top + 200

    print(date, 'sink channels', sink_channels)
    print(date, 'sink_max', sink_max)
    print(date, 'l4 top', l4_top, 'um')
    print(date, 'l4 bottom', l4_bottom, 'um')
    
    l4 = ((unit_depths < l4_top) & (unit_depths > l4_bottom))

    l23 = ((unit_depths < l23_top) & (unit_depths > l23_bottom))
    
    return l4_top, l4_bottom, l4, l23_top, l23_bottom, l23




def perform_opto_tag(data, no_bins, resp_window, trial_length, response_bin):
    
    ''' calculates optogenetic responses and performs putative optotagging
        based on % responsive trials
        
        TODO - incorporate latency and response measures
        
        inputs: "data" - neurons x trials x spike times 
        
                "no_bins" - how many bins? 3000 for 1ms bins in 3s trial
                
                "resp window" - how long after stim to count responses - 20ms usually
                
                "trial length" - usually 3s for optotag
                
        outputs: "opto_trial_counts" - neurons x trials - mean binned spike counts for each neuron
        
                "opto_responses" - % of trials with response in window
                
                "opto_tag" - boolean of opto_tagged neurons
                
        Matt Buchan // Akerman Lab - Aug 2020
                
                '''
    import numpy as np

    opto_trial_counts = []
    opto_resp_perc = []
    opto_latency = []
    opto_spont = []

    for neuron in range(len(data)):

        spike_times = data[neuron]

        unit_trial_count = []

        unit_response = 0

        unit_latency = []

        unit_spont = []

        for trial in range(len(data[neuron])):

            hist, bins = np.histogram(spike_times[trial], no_bins, range = (0,trial_length))

            spont = sum(hist[0:500]/0.5)

            unit_spont.append(spont)

            if sum(hist[1000:1000+resp_window]) > 0:

                unit_response += 1

            unit_trial_count.append(hist)

            if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.099)).sum() > 0:
            #if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.01)).sum() > 0:


                latency = min(i for i in spike_times[trial] if i > 1)

                latency = latency - 1

            else: 

                latency = float('NaN')

            unit_latency.append(latency)

   #     unit_latency = np.array(unit_latency)[(np.array(unit_latency) < 0.02) & (np.array(unit_latency) > 0.002)].tolist() 
        unit_latency = np.array(unit_latency)[(np.array(unit_latency) < 0.02) & (np.array(unit_latency) > 0.002)].tolist() 


        opto_trial_counts.append(np.nanmean(unit_trial_count, axis = 0))
        opto_resp_perc.append(unit_response)

        opto_spont.append(np.array(unit_spont))

        if np.array(unit_latency).sum() > 0: 

            opto_latency.append(np.mean(unit_latency,0))
           # opto_latency.append(min(unit_latency))

        else: 

            opto_latency.append(float('Nan'))



    window_start = 1000
    window_stop = window_start + response_bin 

    opto_bin_responses = [np.sum(resp[window_start:window_stop]) for resp in opto_trial_counts]
    
    opto_tag = ((np.asarray(opto_resp_perc) >= 15) | (np.asarray(opto_bin_responses) >= 1))

    return opto_trial_counts, opto_resp_perc, opto_tag, opto_bin_responses, opto_latency, opto_spont


def split_units(opto_tag, rs_units, fs_units):

    opto_rs_units = ((opto_tag == True) & (rs_units == True))
    opto_fs_units = ((opto_tag == True) & (fs_units == True))

    return opto_rs_units, opto_fs_units


def generate_labels(unit_depths):

    import numpy as np

    labels = []

    [labels.append(np.random.randint(111111, 999999)) for unit in unit_depths]

    return labels

def generate_dates(date, unit_depths):

    import numpy as np

    dates = []

    [dates.append(date) for unit in unit_depths]

    return dates