def build_dfs(date, data_type):

    import seaborn as sns
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    sns.set_context('talk')

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

    opto_trial_counts, opto_resp_perc, opto_tag, opto_bin_responses, opto_latency, opto_spont = up.perform_opto_tag(opto_tag_10, 3000, 20, 3, 50)

    #print('optotag complete')

    # split units into fs and rs opto tags

    opto_rs_units, opto_fs_units = up.split_units(opto_tag, rs_units, fs_units)

    #print('unit split complete')

    # perform coupling analysis

    population_stpr, population_stlfp, unit_mua_coupling, unit_lfp_coupling, stpr_1st_half, stpr_2nd_half = spont.calculate_coupling(spont_spikes, spont_lfp, spont_mua_spikes, 'single_unit')

    # import matplotlib.pyplot as plt

    # [plt.plot(stpr) for stpr in population_stpr[rs_units]]

    # plt.savefig(os.path.join(figsave, date, 'rs_stpr.svg'))
    # plt.savefig(os.path.join(figsave, date, 'rs_stpr.png'))

    #perform single whisk analysis

    pw_ID, pw_trial_counts, pw_resp_perc, pw_latency, pw_bin_responses, aw_trial_counts, aw_resp_perc, aw_latency, aw_bin_responses, w1_avg_response, w2_avg_response, big_spont_responses = whisk.dual_whisk_single_analysis(single_whisk_1, single_whisk_2)
    
    if date == "2020_06_24_1":

       pw_ID = 1
   
    # Perform quad whisk analysis

    pw_quad_trial_counts, aw_quad_trial_counts, pw_1_latency, aw_1_latency, pw_quad_1, pw_quad_2, pw_quad_3, pw_quad_4, aw_quad_1, aw_quad_2, aw_quad_3, aw_quad_4, pw_ratio_2_1, pw_ratio_4_1, aw_ratio_2_1, aw_ratio_4_1  = whisk.dual_whisk_quad_analysis(pw_ID, quad_whisk_1, quad_whisk_2, w1_avg_response, w2_avg_response)

    #perform frequency analysis

    #pw_freq_counts, aw_freq_counts, pw_freq_resps, aw_freq_resps, pw_freq_latency, aw_freq_latency = whisk.frequency_analysis(freq_spikes_1, freq_spikes_2, w1_avg_response, w2_avg_response)

    ## calculate CSD profile from electrode depths

    csd = up.calculate_csd(lfp_1, lfp_2, 1, pw_ID)

    # plot CSD profile ## TO DO ## make plot nicer

    #up.plot_csd(csd, 100, date)

    # split units into layers 

    l4_top, l4_bottom, l4, l23_top, l23_bottom, l23 = up.calculate_l4(csd, unit_depths, date, 'spec')

    time = np.arange(0,25,1)

    if pw_ID == 1: 

       w1_autocorr, w1_tau, w1_fit = whisk.intrinsic_timescale(single_whisk_1, time)
    
    if pw_ID == 2:

       w1_autocorr, w1_tau, w1_fit = whisk.intrinsic_timescale(single_whisk_2, time)

    pw_peak_latency, aw_peak_latency = whisk.compute_peak_latency(pw_trial_counts, aw_trial_counts)

#build dataframe

    data = {'date': dates, 'label': labels, 'depths': unit_depths, 'rs': rs_units, 'fs': fs_units, 
           'opto_rs': opto_rs_units, 'opto_fs': opto_fs_units, 't2p': pop_t2p,
           'half_width': pop_half_width[:,0], 'l4': l4, 'l23': l23, 'mua_coupling': unit_mua_coupling,
           'lfp_coupling': unit_lfp_coupling, 'pw_resp_perc': pw_resp_perc, 'pw_latency': pw_latency, 'pw_1_latency': pw_1_latency, 
           'pw_bin_resp': pw_bin_responses, 'aw_resp_perc': aw_resp_perc, 'aw_latency': aw_latency, 'aw_1_latency': aw_1_latency,
           'aw_bin_resp': aw_bin_responses, 'pw_1': pw_quad_1, 'pw_2': pw_quad_2, 'pw_3': pw_quad_3,
           'pw_4': pw_quad_4, 'aw_1': aw_quad_1, 'aw_2': aw_quad_2, 'aw_3': aw_quad_3,'aw_4': aw_quad_4,
           'opto_resp_perc': opto_resp_perc, 'opto_bin_resp': opto_bin_responses, 'spont_resp': big_spont_responses,
           'stpr_1st': stpr_1st_half, 'stpr_2nd': stpr_2nd_half, 'entropy': unit_entropy, 'opto_latency': opto_latency,
           'opto_spont': opto_spont, 'pw_trial_counts':pw_trial_counts, 'aw_trial_counts': aw_trial_counts, 
           'pw_quad_trial_counts': pw_quad_trial_counts, 'aw_quad_trial_counts': aw_quad_trial_counts, 'opto_trial_counts': opto_trial_counts,
           'autocorr': w1_autocorr.tolist(), 'tau': w1_tau.tolist(), 'fit': w1_fit.tolist(), 'pw_peak_latency': pw_peak_latency, 'aw_peak_latency': aw_peak_latency}


    import pandas as pd

    df = pd.DataFrame(data)

    df.to_pickle(os.path.join(save, '{}.pkl'.format(date))) 

