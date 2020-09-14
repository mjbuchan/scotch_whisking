def build_dfs(date):

    import seaborn as sns
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    sns.set_context('talk')

    import scotch_whisking.whisk_analysis as whisk
    import scotch_whisking.unit_preprocessing as up
    import scotch_whisking.spont_analysis as spont

    # set paths

    path = '/Volumes/Seagate Expansion Drive/PS2_backup/ChR2_ON/data'

    save = '/Volumes/Seagate Expansion Drive/PS2_backup/ChR2_ON/processed_dfs'

    figsave = '/Volumes/Seagate Expansion Drive/PS2_backup/ChR2_ON/figures'

    # load data

    unit_waveforms, lfp, opto_tag_10, unit_depths, spont_spikes, spont_lfp, spont_mua_spikes, single_whisk_1, single_whisk_2, quad_whisk_1, quad_whisk_2 = up.load_data(date, path)

    labels = up.generate_labels(date, unit_depths)

    # calculate waveform properties

    pop_t2p, pop_trough_val, pop_peak_val, pop_half_width = up.waveform_analysis(unit_waveforms)

    print('waveform analysis complete')

    # cluster into putative fs and rs units

    fs_units, rs_units = up.cluster_units(pop_t2p, pop_half_width)

    print('clustering complete')

    # plot kde of clusters with arbitrary .55 cut-off

    if fs_units.sum() > 1:

        up.plot_unit_clusters(pop_t2p, pop_half_width, fs_units, rs_units)
        
    else:
        
        print('not enough fs units for plot')

    print('fs units:', fs_units.sum())
    print('rs units:', rs_units.sum())

    plt.savefig(os.path.join(figsave, date, 'clustered_units.svg'))
    plt.savefig(os.path.join(figsave, date, 'clustered_units.png'))

    ## calculate CSD profile from electrode depths

    csd, csd_smooth, depth = up.calculate_csd(lfp, 15)

    # plot CSD profile ## TO DO ## make plot nicer

    up.plot_csd(csd_smooth, 40, 100)

    # split units into layers 

    l4_top, l4_bottom, l4 = up.calculate_l4(csd_smooth, depth, unit_depths)

    print('layer split complete')

    plt.savefig(os.path.join(figsave, date, 'csd.svg'))
    plt.savefig(os.path.join(figsave, date, 'csd.png'))

    # perform optotagging

    opto_trial_counts, opto_resp_perc, opto_tag, opto_bin_responses = up.perform_opto_tag(opto_tag_10, 3000, 20, 3, 50)

    print('optotag complete')

    # split units into fs and rs opto tags

    opto_rs_units, opto_fs_units = up.split_units(opto_tag, rs_units, fs_units)

    print('unit split complete')

    # perform coupling analysis

    population_stpr, population_stlfp, unit_mua_coupling, unit_lfp_coupling, stpr_1st_half, stpr_2nd_half = spont.calculate_coupling(spont_spikes, spont_lfp, spont_mua_spikes, 'single_unit')

    import matplotlib.pyplot as plt

    [plt.plot(stpr) for stpr in population_stpr[rs_units]]

    plt.savefig(os.path.join(figsave, date, 'rs_stpr.svg'))
    plt.savefig(os.path.join(figsave, date, 'rs_stpr.png'))

    #perform single whisk analysis

    pw_trial_counts, pw_resp_perc, pw_latency, pw_bin_responses, aw_trial_counts, aw_resp_perc, aw_latency, aw_bin_responses, w1_avg_response, w2_avg_response = whisk.dual_whisk_single_analysis(single_whisk_1, single_whisk_2)

    # Perform quad whisk analysis

    pw_quad_trial_counts, aw_quad_trial_counts, pw_quad_1, pw_quad_2, pw_quad_3, pw_quad_4, aw_quad_1, aw_quad_2, aw_quad_3, aw_quad_4, pw_ratio_2_1, pw_ratio_4_1, aw_ratio_2_1, aw_ratio_4_1  = whisk.dual_whisk_quad_analysis(quad_whisk_1, quad_whisk_2, w1_avg_response, w2_avg_response)

    plt.figure(figsize = (4,4))

    plt.plot(np.mean(pw_trial_counts,0))

    plt.xlabel('Time (s)')
    plt.ylabel('Spike probability')

    plt.savefig(os.path.join(figsave, date, 'pw_resp.svg'))
    plt.savefig(os.path.join(figsave, date, 'pw_resp.png'))

    plt.figure(figsize = (4,4))

    plt.plot(np.mean(aw_trial_counts,0))

    plt.xlabel('Time (s)')
    plt.ylabel('Spike probability')

    plt.savefig(os.path.join(figsave, date, 'aw_resp.svg'))
    plt.savefig(os.path.join(figsave, date, 'aw_resp.png'))

    plt.figure(figsize = (4,4))

    plt.plot(np.mean(pw_quad_trial_counts,0))

    plt.xlabel('Time (s)')
    plt.ylabel('Spike probability')

    plt.savefig(os.path.join(figsave, date, 'pw_quad_resp.svg'))
    plt.savefig(os.path.join(figsave, date, 'pw_quad_resp.png'))

    plt.figure(figsize = (4,4))

    plt.plot(np.mean(aw_quad_trial_counts,0))

    plt.xlabel('Time (s)')
    plt.ylabel('Spike probability')

    plt.savefig(os.path.join(figsave, date, 'aw_quad_resp.svg'))
    plt.savefig(os.path.join(figsave, date, 'aw_quad_resp.png'))

    data = {'label': labels, 'depths': unit_depths, 'rs': rs_units, 'fs': fs_units, 
           'opto_rs': opto_rs_units, 'opto_fs': opto_fs_units, 't2p': pop_t2p,
           'half_width': pop_half_width[:,0], 'l4': l4, 'mua_coupling': unit_mua_coupling,
           'lfp_coupling': unit_lfp_coupling, 'pw_resp_perc': pw_resp_perc, 'pw_latency': pw_latency,
           'pw_bin_resp': pw_bin_responses, 'aw_resp_perc': aw_resp_perc, 'aw_latency': aw_latency,
           'aw_bin_resp': aw_bin_responses, 'pw_1': pw_quad_1, 'pw_2': pw_quad_2, 'pw_3': pw_quad_3,
           'pw_4': pw_quad_4, 'aw_1': aw_quad_1, 'aw_2': aw_quad_2, 'aw_3': aw_quad_3,'aw_4': aw_quad_4,
           'opto_resp_perc': opto_resp_perc, 'opto_bin_resp': opto_bin_responses}


import pandas as pd

df = pd.DataFrame(data)

    df.to_pickle(os.path.join(save, '{}.pkl'.format(date))) 

