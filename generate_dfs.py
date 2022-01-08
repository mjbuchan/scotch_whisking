def build_dfs(date, data_type):

    import seaborn as sns
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    sns.set_context('talk')
    from scipy.io import loadmat

    import scotch_whisking.whisk_analysis as whisk
    import scotch_whisking.unit_preprocessing as up
    import scotch_whisking.spont_analysis as spont

    # set paths

    if data_type == 'chr2_on':

       path = '/Users/matthewbuchan/Desktop/data_backup/in_vivo/chr2_on/data'

       save = '/Users/matthewbuchan/Desktop/data_backup/in_vivo/chr2_on/processed_dfs'

       figsave = '/Users/matthewbuchan/Desktop/data_backup/in_vivo/chr2_on/figures'

    if data_type == 'cag_lhx2':

       path = '/Users/matthewbuchan/Desktop/data_backup/in_vivo/cag_lhx2/data'

       save = '/Users/matthewbuchan/Desktop/data_backup/in_vivo/cag_lhx2/processed_dfs'

       figsave = '/Users/matthewbuchan/Desktop/data_backup/in_vivo/cag_lhx2/figures'

    if data_type == 'chr2_cag':

       path = '/Users/matthewbuchan/Desktop/data_backup/in_vivo/chr2_cag/data'

       save = '/Users/matthewbuchan/Desktop/data_backup/in_vivo/chr2_cag/processed_dfs'

       figsave = '/Users/matthewbuchan/Desktop/data_backup/in_vivo/chr2_cag/figures'

    # load data

    unit_waveforms, lfp_1, lfp_2, opto_tag_10, unit_depths, spont_spikes, spont_lfp, spont_mua_spikes, single_whisk_1, single_whisk_2, quad_whisk_1, quad_whisk_2 = up.load_data(date, path)

    labels = up.generate_labels(unit_depths)
    dates = up.generate_dates(date, unit_depths)

    #freq_spikes_1, freq_spikes_2 = up.load_freq_data(path,date)

    # calculate waveform properties

    pop_t2p, pop_trough_val, pop_peak_val, pop_half_width = up.waveform_analysis(unit_waveforms)

    #print('waveform analysis complete')

    # cluster into putative fs and rs units

    fs_units, rs_units = up.cluster_units(pop_t2p, pop_half_width)

    print('clustering complete')

    #plot kde of clusters with arbitrary .55 cut-off

   # if fs_units.sum() > 1:

   #     up.plot_unit_clusters(pop_t2p, pop_half_width, fs_units, rs_units)
        
   # else:
        
   #     print('not enough fs units for plot')

    print('fs units:', fs_units.sum())
    print('rs units:', rs_units.sum())

   # plt.savefig(os.path.join(figsave, date, 'clustered_units.svg'))
   # plt.savefig(os.path.join(figsave, date, 'clustered_units.png'))

    # calculate entropy

    unit_entropy = spont.calculate_unit_entropy(spont_spikes)

    # perform optotagging

    opto_trial_counts, opto_resp_perc, opto_tag, opto_bin_responses, opto_latency, opto_spont, opto_jitter = up.perform_opto_tag(opto_tag_10, 3000, 20, 3, 50)

    #print('optotag complete')

    # split units into fs and rs opto tags

    opto_rs_units, opto_fs_units = up.split_units(opto_tag, rs_units, fs_units)

    #print('unit split complete')

    # perform coupling analysis

    population_stpr, population_stlfp, unit_mua_coupling, unit_lfp_coupling, stpr_1st_half, stpr_2nd_half = spont.calculate_coupling(spont_spikes, spont_lfp, spont_mua_spikes, 'single_unit')

    population_strip = []

    [population_strip.append(i) for i in population_stpr]

    population_stpr = population_strip
    # import matplotlib.pyplot as plt

    # [plt.plot(stpr) for stpr in population_stpr[rs_units]]

    # plt.savefig(os.path.join(figsave, date, 'rs_stpr.svg'))
    # plt.savefig(os.path.join(figsave, date, 'rs_stpr.png'))

    #perform single whisk analysis

    pw_ID, pw_trial_counts, pw_resp_perc, pw_latency, pw_bin_responses, aw_trial_counts, aw_resp_perc, aw_latency, aw_bin_responses, w1_avg_response, w2_avg_response, big_spont_responses = whisk.dual_whisk_single_analysis(single_whisk_1, single_whisk_2)
    
    if date == "2020_06_24_1":

       pw_ID = 1
   
    # Perform quad whisk analysis

    pw_quad_trial_counts, aw_quad_trial_counts, pw_1_latency, aw_1_latency, pw_quad_1, pw_quad_2, pw_quad_3, pw_quad_4, aw_quad_1, aw_quad_2, aw_quad_3, aw_quad_4, pw_ratio_2_1, pw_ratio_4_1, aw_ratio_2_1, aw_ratio_4_1, pw_total, aw_total  = whisk.dual_whisk_quad_analysis(pw_ID, quad_whisk_1, quad_whisk_2, w1_avg_response, w2_avg_response)

   
    ## calculate CSD profile from electrode depths

    csd = up.calculate_csd(lfp_1, lfp_2, 1, pw_ID)

    # plot CSD profile ## TO DO ## make plot nicer

    #up.plot_csd(csd, 100, date)

    # split units into layers 

    l4_top, l4_bottom, l4, l23_top, l23_bottom, l23 = up.calculate_l4(csd, unit_depths, date, 'spec')

    time = np.arange(0,300,1)

    if pw_ID == 1: 

       w1_autocorr, w1_tau, w1_fit = whisk.intrinsic_timescale(quad_whisk_1, time)
    
    if pw_ID == 2:

       w1_autocorr, w1_tau, w1_fit = whisk.intrinsic_timescale(quad_whisk_2, time)

    if pw_ID == 1: 

       w2_autocorr, w2_tau, w2_fit = whisk.intrinsic_timescale(quad_whisk_2, time)
    
    if pw_ID == 2:

       w2_autocorr, w2_tau, w2_fit = whisk.intrinsic_timescale(quad_whisk_1, time)

    pw_peak_latency, aw_peak_latency = whisk.compute_peak_latency(pw_trial_counts, aw_trial_counts)


   #  experiment = 'frequency'

   #  spikes_1_4 = loadmat(os.path.join(path, date, experiment, 'spikes_1_4'))
   #  spikes_2_4 = loadmat(os.path.join(path, date, experiment, 'spikes_2_4'))

   #  spikes_1_4 = spikes_1_4['spikes_1_4']
   #  spikes_2_4 = spikes_2_4['spikes_2_4']

   #  spikes_1_8 = loadmat(os.path.join(path, date, experiment, 'spikes_1_8'))
   #  spikes_2_8 = loadmat(os.path.join(path, date, experiment, 'spikes_2_8'))

   #  spikes_1_8 = spikes_1_8['spikes_1_8']
   #  spikes_2_8 = spikes_2_8['spikes_2_8']

   #  spikes_1_16 = loadmat(os.path.join(path, date, experiment, 'spikes_1_16'))
   #  spikes_2_16 = loadmat(os.path.join(path, date, experiment, 'spikes_2_16'))

   #  spikes_1_16 = spikes_1_16['spikes_1_16']
   #  spikes_2_16 = spikes_2_16['spikes_2_16']

   #  lfp_1_4 = loadmat(os.path.join(path, date, experiment, 'lfp_1_4'))
   #  lfp_2_4 = loadmat(os.path.join(path, date, experiment, 'lfp_2_4'))

   #  lfp_1_4 = np.mean(lfp_1_4['lfp_1_4'][8:16],0)
   #  lfp_2_4 = np.mean(lfp_2_4['lfp_2_4'][8:16],0)

   #  lfp_1_8 = loadmat(os.path.join(path, date, experiment, 'lfp_1_8'))
   #  lfp_2_8 = loadmat(os.path.join(path, date, experiment, 'lfp_2_8'))

   #  lfp_1_8 = np.mean(lfp_1_8['lfp_1_8'][8:16],0)
   #  lfp_2_8 = np.mean(lfp_2_8['lfp_2_8'][8:16],0)

   #  lfp_1_16 = loadmat(os.path.join(path, date, experiment, 'lfp_1_16'))
   #  lfp_2_16 = loadmat(os.path.join(path, date, experiment, 'lfp_2_16'))

   #  lfp_1_16 = np.mean(lfp_1_16['lfp_1_16'][8:16],0)
   #  lfp_2_16 = np.mean(lfp_2_16['lfp_2_16'][8:16],0)
   #  #4Hz

   #  hilbert_1_real, hilbert_1_imag = whisk.transform_lfp(lfp_1_4, 3, 5)
   #  hilbert_2_real, hilbert_2_imag = whisk.transform_lfp(lfp_2_4, 3, 5)

   #  pw_ppc_4, aw_ppc_4 = whisk.whisk_phase(pw_ID, spikes_1_4, hilbert_1_real, hilbert_1_imag, spikes_2_4, hilbert_2_real, hilbert_2_imag)

   # #8Hz

   #  hilbert_1_real, hilbert_1_imag = whisk.transform_lfp(lfp_1_8, 7, 9)
   #  hilbert_2_real, hilbert_2_imag = whisk.transform_lfp(lfp_2_8, 7, 9)

   #  pw_ppc_8, aw_ppc_8 = whisk.whisk_phase(pw_ID, spikes_1_8, hilbert_1_real, hilbert_1_imag, spikes_2_8, hilbert_2_real, hilbert_2_imag)

   # #16Hz

   #  hilbert_1_real, hilbert_1_imag = whisk.transform_lfp(lfp_1_16, 15, 17)
   #  hilbert_2_real, hilbert_2_imag = whisk.transform_lfp(lfp_2_16, 15, 17)

   #  pw_ppc_16, aw_ppc_16 = whisk.whisk_phase(pw_ID, spikes_1_16, hilbert_1_real, hilbert_1_imag, spikes_2_16, hilbert_2_real, hilbert_2_imag)

   #  if pw_ID == 1:

   #     aw_trace_4, aw_train_4, aw_idx_4, fr_aw_4 = whisk.quick_stim_freq(spikes_2_4, 4)
   #     pw_trace_4, pw_train_4, pw_idx_4, fr_pw_4 = whisk.quick_stim_freq(spikes_1_4, 4)

   #     aw_trace_8, aw_train_8, aw_idx_8, fr_aw_8 = whisk.quick_stim_freq(spikes_2_8, 8)
   #     pw_trace_8, pw_train_8, pw_idx_8, fr_pw_8 = whisk.quick_stim_freq(spikes_1_8, 8)

   #     aw_trace_16, aw_train_16, aw_idx_16, fr_aw_16 = whisk.quick_stim_freq(spikes_2_16, 16)
   #     pw_trace_16, pw_train_16, pw_idx_16, fr_pw_16 = whisk.quick_stim_freq(spikes_1_16, 16)

   #  if pw_ID == 2:

   #     aw_trace_4, aw_train_4, aw_idx_4, fr_aw_4  = whisk.quick_stim_freq(spikes_1_4, 4)
   #     pw_trace_4, pw_train_4, pw_idx_4, fr_pw_4 = whisk.quick_stim_freq(spikes_2_4, 4)

   #     aw_trace_8, aw_train_8, aw_idx_8, fr_aw_8 = whisk.quick_stim_freq(spikes_1_8, 8)
   #     pw_trace_8, pw_train_8, pw_idx_8, fr_pw_8 = whisk.quick_stim_freq(spikes_2_8, 8)

   #     aw_trace_16, aw_train_16, aw_idx_16, fr_aw_16 = whisk.quick_stim_freq(spikes_1_16, 16)
   #     pw_trace_16, pw_train_16, pw_idx_16, fr_pw_16 = whisk.quick_stim_freq(spikes_2_16, 16)
#build dataframe

    data = {'date': dates, 'label': labels, 'depths': unit_depths, 'rs': rs_units, 'fs': fs_units, 
           'opto_rs': opto_rs_units, 'opto_fs': opto_fs_units, 't2p': pop_t2p,
           'half_width': pop_half_width[:,0], 'l4': l4, 'l23': l23, 'mua_coupling': unit_mua_coupling,
           'lfp_coupling': unit_lfp_coupling, 'stpr': population_stpr, 'pw_resp_perc': pw_resp_perc, 'pw_latency': pw_latency, 'pw_1_latency': pw_1_latency, 
           'pw_bin_resp': pw_bin_responses, 'aw_resp_perc': aw_resp_perc, 'aw_latency': aw_latency, 'aw_1_latency': aw_1_latency,
           'aw_bin_resp': aw_bin_responses, 'pw_1': pw_quad_1, 'pw_2': pw_quad_2, 'pw_3': pw_quad_3,
           'pw_4': pw_quad_4, 'aw_1': aw_quad_1, 'aw_2': aw_quad_2, 'aw_3': aw_quad_3,'aw_4': aw_quad_4,
           'opto_resp_perc': opto_resp_perc, 'opto_bin_resp': opto_bin_responses, 'spont_resp': big_spont_responses,
           'stpr_1st': stpr_1st_half, 'stpr_2nd': stpr_2nd_half, 'entropy': unit_entropy, 'opto_latency': opto_latency,
           'opto_spont': opto_spont, 'pw_trial_counts':pw_trial_counts, 'aw_trial_counts': aw_trial_counts, 
           'pw_quad_trial_counts': pw_quad_trial_counts, 'aw_quad_trial_counts': aw_quad_trial_counts, 'opto_trial_counts': opto_trial_counts,
           'w1_autocorr': w1_autocorr.tolist(), 'w1_tau': w1_tau.tolist(), 'w1_fit': w1_fit.tolist(), 'w2_autocorr': w2_autocorr.tolist(), 'w2_tau': w2_tau.tolist(), 'w2_fit': w2_fit.tolist(),'pw_peak_latency': pw_peak_latency, 'aw_peak_latency': aw_peak_latency,
           'pw_total': pw_total, 'aw_total': aw_total, 'opto_jitter': opto_jitter}

         #   'aw_trace_4': aw_trace_4, 'aw_train_4': aw_train_4, 'aw_idx_4': aw_idx_4, 'pw_trace_4': pw_trace_4, 'pw_train_4': pw_train_4, 'pw_idx_4': pw_idx_4, 
         #   'aw_trace_8': aw_trace_8, 'aw_train_8': aw_train_8, 'aw_idx_8': aw_idx_8, 'pw_trace_8': pw_trace_8, 'pw_train_8': pw_train_8, 'pw_idx_8': pw_idx_8,
         #   'aw_trace_16': aw_trace_16, 'aw_train_16': aw_train_16, 'aw_idx_16': aw_idx_16, 'pw_trace_16': pw_trace_16, 'pw_train_16': pw_train_16, 'pw_idx_16': pw_idx_16,
         #   'fr_aw_4': fr_aw_4, 'fr_pw_4': fr_pw_4, 'fr_aw_8': fr_aw_8, 'fr_pw_8': fr_pw_8, 'fr_aw_16': fr_aw_16, 'fr_pw_16': fr_pw_16,
         #   'pw_ppc_4': pw_ppc_4, 'aw_ppc_4': aw_ppc_4, 'pw_ppc_8': pw_ppc_8, 'aw_ppc_8': aw_ppc_8, 'pw_ppc_16': pw_ppc_16, 'aw_ppc_16': aw_ppc_16}


    import pandas as pd

    df = pd.DataFrame(data)

    df.to_pickle(os.path.join(save, '{}.pkl'.format(date))) 

