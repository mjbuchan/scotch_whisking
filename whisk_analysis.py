def dual_whisk_single_analysis(whisk_1, whisk_2):

    ''' Hella script to spit out single whisk details - 
        importantly works out which whisker is which

        Inputs just whisk_1 and whisk_2 = neurons x trials x spike times arrays

        Outputs everything...


        Matt Buchan // Akerman Lab - Sept 2020
    '''

    import numpy as np 

    w1_trial_counts = []
    w1_resp_perc = []
    w1_latency = []

    for neuron in range(len(whisk_1)):

        spike_times = whisk_1[neuron]

        unit_trial_count = []

        unit_response = 0

        unit_latency = []

        for trial in range(len(whisk_1[neuron])):

            hist, bins = np.histogram(spike_times[trial], 3000, range = (0,3))

            if sum(hist[1000:1050]) > 0:

                unit_response += 1

            unit_trial_count.append(hist)

            if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.099)).sum() > 0:

                latency = min(i for i in spike_times[trial] if i > 1)

                latency = latency - 1

            else: 

                latency = float('NaN')

            unit_latency.append(latency)

        w1_trial_counts.append(np.nanmean(unit_trial_count, axis = 0))
        w1_resp_perc.append(unit_response)
        w1_latency.append(np.nanmean(unit_latency))

    w2_trial_counts = []
    w2_resp_perc = []
    w2_latency = []

    for neuron in range(len(whisk_2)):

        spike_times = whisk_2[neuron]

        unit_trial_count = []

        unit_response = 0

        unit_latency = []

        for trial in range(len(whisk_2[neuron])):

            hist, bins = np.histogram(spike_times[trial], 3000, range = (0,3))

            if sum(hist[1000:1050]) > 0:

                unit_response += 1

            unit_trial_count.append(hist)

            if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.05)).sum() > 0:

                latency = min(i for i in spike_times[trial] if i > 1)

                latency = latency - 1

            else: 

                latency = float('NaN')

            unit_latency.append(latency)

        w2_trial_counts.append(np.nanmean(unit_trial_count, axis = 0))
        w2_resp_perc.append(unit_response)
        w2_latency.append(np.nanmean(unit_latency, 0))

    w1_bin_responses = [np.sum(resp[1000:1050]) for resp in w1_trial_counts]
    w1_spont_responses = [np.sum(resp[900:950]) for resp in w1_trial_counts]
    w1_bin_responses = (np.array(w1_bin_responses) - np.array(w1_spont_responses)).tolist()

    w1_avg_response = np.mean(w1_bin_responses)

    w2_bin_responses = [np.sum(resp[1000:1050]) for resp in w2_trial_counts]
    w2_spont_responses = [np.sum(resp[900:950]) for resp in w2_trial_counts]
    w2_bin_responses = (np.array(w2_bin_responses) - np.array(w2_spont_responses)).tolist()

    w2_avg_response = np.mean(w2_bin_responses)

    if w1_avg_response > w2_avg_response:

        pw_trial_counts = w1_trial_counts
        pw_resp_perc = w1_resp_perc
        pw_latency = w1_latency
        pw_bin_responses = w1_bin_responses

        aw_trial_counts = w2_trial_counts
        aw_resp_perc = w2_resp_perc
        aw_latency = w2_latency
        aw_bin_responses = w2_bin_responses

    if w1_avg_response < w2_avg_response:

        pw_trial_counts = w2_trial_counts
        pw_resp_perc = w2_resp_perc
        pw_latency = w2_latency
        pw_bin_responses = w2_bin_responses

        aw_trial_counts = w1_trial_counts
        aw_resp_perc = w1_resp_perc
        aw_latency = w1_latency
        aw_bin_responses = w1_bin_responses

    return pw_trial_counts, pw_resp_perc, pw_latency, pw_bin_responses, aw_trial_counts, aw_resp_perc, aw_latency, aw_bin_responses, w1_avg_response, w2_avg_response


def dual_whisk_quad_analysis(whisk_1, whisk_2, w1_avg_response, w2_avg_response):

    ''' Hella script to spit out quad whisk details - 
        importantly works out which whisker is which

        Inputs just quad_whisk_1 and quad_whisk_2 = neurons x trials x spike times arrays
        plus you also need the avg responses from the single analysis to be consistent on pw/aw
        
        
        Outputs everything...


        Matt Buchan // Akerman Lab - Sept 2020
    '''

    import numpy as np 

    w1_trial_counts = []
    w1_resp_perc = []
    w1_latency = []

    for neuron in range(len(whisk_1)):

        spike_times = whisk_1[neuron]

        unit_trial_count = []

        unit_response = 0

        unit_latency = []

        for trial in range(len(whisk_1[neuron])):

            hist, bins = np.histogram(spike_times[trial], 3000, range = (0,3))

            if sum(hist[1000:1050]) > 0:

                unit_response += 1

            unit_trial_count.append(hist)

            if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.05)).sum() > 0:

                latency = min(i for i in spike_times[trial] if i > 1)

                latency = latency - 1

            else: 

                latency = float('NaN')

            unit_latency.append(latency)

        w1_trial_counts.append(np.nanmean(unit_trial_count, axis = 0))
        w1_resp_perc.append(unit_response)
        w1_latency.append(np.nanmean(unit_latency))

    w2_trial_counts = []
    w2_resp_perc = []
    w2_latency = []

    for neuron in range(len(whisk_2)):

        spike_times = whisk_2[neuron]

        unit_trial_count = []

        unit_response = 0

        unit_latency = []

        for trial in range(len(whisk_2[neuron])):

            hist, bins = np.histogram(spike_times[trial], 3000, range = (0,3))

            if sum(hist[1000:1050]) > 0:

                unit_response += 1

            unit_trial_count.append(hist)

            if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.05)).sum() > 0:

                latency = min(i for i in spike_times[trial] if i > 1)

                latency = latency - 1

            else: 

                latency = float('NaN')

            unit_latency.append(latency)

        w2_trial_counts.append(np.nanmean(unit_trial_count, axis = 0))
        w2_resp_perc.append(unit_response)
        w2_latency.append(np.nanmean(unit_latency, 0))

    w1_resp_1 = [np.sum(resp[1000:1050]) for resp in w1_trial_counts]
    w1_resp_2 = [np.sum(resp[1100:1150]) for resp in w1_trial_counts]
    w1_resp_3 = [np.sum(resp[1200:1250]) for resp in w1_trial_counts]
    w1_resp_4 = [np.sum(resp[1300:1350]) for resp in w1_trial_counts]
    
    w1_spont_responses = [np.sum(resp[900:950]) for resp in w1_trial_counts]
    
    w1_resp_1 = np.array(w1_resp_1) - np.array(w1_spont_responses).tolist()
    w1_resp_2 = np.array(w1_resp_2) - np.array(w1_spont_responses).tolist()
    w1_resp_3 = np.array(w1_resp_3) - np.array(w1_spont_responses).tolist()
    w1_resp_4 = np.array(w1_resp_4) - np.array(w1_spont_responses).tolist()
    
    w1_resp_1 = np.where(w1_resp_1<0, 0.001, w1_resp_1)
    w1_resp_2 = np.where(w1_resp_2<0, 0.001, w1_resp_2)
    w1_resp_3 = np.where(w1_resp_3<0, 0.001, w1_resp_3)
    w1_resp_4 = np.where(w1_resp_4<0, 0.001, w1_resp_4)
    
    w2_resp_1 = [np.sum(resp[1000:1050]) for resp in w2_trial_counts]
    w2_resp_2 = [np.sum(resp[1100:1150]) for resp in w2_trial_counts]
    w2_resp_3 = [np.sum(resp[1200:1250]) for resp in w2_trial_counts]
    w2_resp_4 = [np.sum(resp[1300:1350]) for resp in w2_trial_counts]
    
    w2_spont_responses = [np.sum(resp[900:950]) for resp in w2_trial_counts]
    
    w2_resp_1 = np.array(w2_resp_1) - np.array(w2_spont_responses).tolist()
    w2_resp_2 = np.array(w2_resp_2) - np.array(w2_spont_responses).tolist()
    w2_resp_3 = np.array(w2_resp_3) - np.array(w2_spont_responses).tolist()
    w2_resp_4 = np.array(w2_resp_4) - np.array(w2_spont_responses).tolist()
    
    w2_resp_1 = np.where(w2_resp_1<0, 0.001, w2_resp_1)
    w2_resp_2 = np.where(w2_resp_2<0, 0.001, w2_resp_2)
    w2_resp_3 = np.where(w2_resp_3<0, 0.001, w2_resp_3)
    w2_resp_4 = np.where(w2_resp_4<0, 0.001, w2_resp_4)

    if w1_avg_response > w2_avg_response:

        pw_quad_1 = w1_resp_1
        pw_quad_2 = w1_resp_2
        pw_quad_3 = w1_resp_3
        pw_quad_4 = w1_resp_4
        
        aw_quad_1 = w2_resp_1
        aw_quad_2 = w2_resp_2
        aw_quad_3 = w2_resp_3
        aw_quad_4 = w2_resp_4
        
        pw_quad_trial_counts = w1_trial_counts
        aw_quad_trial_counts = w2_trial_counts

    if w1_avg_response < w2_avg_response:

        pw_quad_1 = w2_resp_1
        pw_quad_2 = w2_resp_2
        pw_quad_3 = w2_resp_3
        pw_quad_4 = w2_resp_4
        
        aw_quad_1 = w1_resp_1
        aw_quad_2 = w1_resp_2
        aw_quad_3 = w1_resp_3
        aw_quad_4 = w1_resp_4
        
        pw_quad_trial_counts = w2_trial_counts
        aw_quad_trial_counts = w1_trial_counts
    
    pw_ratio_2_1 = (pw_quad_2/pw_quad_1)*100
    pw_ratio_4_1 = (pw_quad_4/pw_quad_1)*100
    
    aw_ratio_2_1 = (aw_quad_2/aw_quad_1)*100
    aw_ratio_4_1 = (aw_quad_4/aw_quad_1)*100

    return pw_quad_trial_counts, aw_quad_trial_counts, pw_quad_1, pw_quad_2, pw_quad_3, pw_quad_4, aw_quad_1, aw_quad_2, aw_quad_3, aw_quad_4, pw_ratio_2_1, pw_ratio_4_1, aw_ratio_2_1, aw_ratio_4_1 