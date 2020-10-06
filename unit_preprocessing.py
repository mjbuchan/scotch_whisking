def load_data(date, path):

    import os
    from scipy.io import loadmat
    import numpy as np
    import scipy.ndimage as nd

    # load waveform data

    unit_waveforms = loadmat(os.path.join(path, date, 'unit_waveforms.mat'))
    unit_waveforms = unit_waveforms['unit_waveforms'] 

    # load lfp data

    lfp = loadmat(os.path.join(path, date, 'whisk_dual_single/lfp_1.mat'))
    lfp = lfp['lfp_1']

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

    return unit_waveforms, lfp, opto_tag_10, unit_depths, spont_spikes, spont_lfp, spont_mua_spikes, single_whisk_1, single_whisk_2, quad_whisk_1, quad_whisk_2

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

    plt.figure(figsize = (4,4))

    sns.kdeplot(pop_t2p[fs_units], pop_half_width[fs_units], cmap = 'Reds', shade = True, shade_lowest = False)
    sns.kdeplot(pop_t2p[rs_units], pop_half_width[rs_units], cmap = 'Blues', shade = True, shade_lowest = False)

    plt.xlim(0, 1.5)
    plt.ylim(0,1)

    plt.xlabel('Trough to peak time (ms)')
    plt.ylabel('Half amplitude duration (ms)')

    plt.vlines(0.5, 0, 1, ls = ':', color = 'grey')

    red = sns.color_palette("Reds")[-2]
    blue = sns.color_palette("Blues")[-2]
    plt.text(0.025, 0.4, "fast-spiking", size=12, color=red)
    plt.text(0.8, 0.8, "regular-spiking", size=12, color=blue)

    plt.tight_layout()


def calculate_csd(lfp, sigma):
    
    ''' code to generate a csd and smoothed csd of 32 channel neuronexus 
        probe lfp sampled at 1000Hz
        
        makes use of elephant resources
        
        inputs: 'lfp' - channels by samples array... currently bit hacky 
                but expects a whisk at 1s, hence the slice from 950:1500
                
                this is converted into neo format for use in elephant csd 
                function
        
                'sigma' - what size of sigma to smooth the csd with. 15 
                works well.
                
        outputs: 'csd' - array of csd profile
                 
                 'csd_smooth' - smoothed array of csd profile for plotting
                 
                 'depth' - depth corresponding to probe locations in um
        
        Matt Buchan mostly stolen from Gemma Gothard // Akerman Lab - Aug 2020
    '''
    
    import elephant
    import neo
    from neo.core import AnalogSignal
    from quantities import ms, s, kHz, uM, uA, mm, um
    
    import numpy as np
    import scipy.ndimage as nd

    print('imports complete')
    
    lfp = np.mean(lfp,axis=1)
    lfp = lfp[:,950:1500].T
    
    c_ind = np.arange(1,33,1)
    c_space = np.arange(0,32*25, 25)
    c_space = c_space[:, np.newaxis]
    
    neo_lfp = AnalogSignal(lfp, units = 'uV', sampling_period = 1*ms, channel_index = c_ind)
    
    print('calculating csd')
    
    csd = elephant.current_source_density.estimate_csd(neo_lfp, coords=c_space*um, method='KCSD1D', process_estimate=True)
    
    csd_smooth = nd.gaussian_filter(csd, sigma = sigma)
    
    depth = np.round(csd.annotations['x_coords']*1000)
    
    return csd, csd_smooth, depth 


def plot_csd(csd, window_min, window_max):

    import matplotlib.pyplot as plt 
    import seaborn as sns
    
    '''
    Plots csd profile within certain window, use 40:100 for 3s single whisk
    trial
    
    Matt Buchan // Akerman Lab - Aug 2020
    '''

    plt.figure(figsize = (2,4))
    sns.heatmap(csd[window_min:window_max].T, cmap = 'coolwarm', cbar_kws={'ticks': []})

    plt.tight_layout()
    plt.yticks([])
    plt.xticks([])


def calculate_l4(csd, depth, unit_depths):

    import numpy as np
    
    ''' calculates likely position of l4 based on 125um boundary on
        either side of the largest current sink, which (in theory) is l4
        
        inputs: "csd" - csd profile calculated by function "calculate csd"
        
                "depth" - depth profile calculated by function "calculate csd"

                "unit_depths" - depths of sorted units
                
        outputs: "l4_top" - top boundary of L4

                 "l4_bottom" - bottom boundary of L4

                 "l4" - boolean mask for sorted units
                 
        Matt Buchan // Akerman Lab - Aug 2020
        '''

    avg_csd = np.mean(csd, 0)
    sink_idx = np.argmin(avg_csd)
    sink_depth = depth[sink_idx]

    l4_top = 800 - sink_depth + 150
    l4_bottom = 800 - sink_depth - 150

    print('l4_top =', l4_top, 'um')
    print('l4_bottom =', l4_bottom, 'um')

    l4 = ((unit_depths < l4_top) & (unit_depths > l4_bottom) )
    
    return l4_top, l4_bottom, l4


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

    for neuron in range(len(data)):

        spike_times = data[neuron]

        unit_trial_count = []

        unit_response = 0

        for trial in range(len(data[neuron])):

            hist, bins = np.histogram(spike_times[trial], no_bins, range = (0,trial_length))

            if sum(hist[1000:1000+resp_window]) > 0:

                unit_response += 1

            unit_trial_count.append(hist)

        opto_trial_counts.append(np.nanmean(unit_trial_count, axis = 0))
        opto_resp_perc.append(unit_response)



    window_start = 1000
    window_stop = window_start + response_bin 

    opto_bin_responses = [np.sum(resp[window_start:window_stop]) for resp in opto_trial_counts]
    
    opto_tag = ((np.asarray(opto_resp_perc) >= 20) | (np.asarray(opto_bin_responses) >= 1))

    return opto_trial_counts, opto_resp_perc, opto_tag, opto_bin_responses


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