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

            #hist, bins = np.histogram(spike_times[trial], 300, range = (0,3))
            hist, bins = np.histogram(spike_times[trial], 3000, range = (0,3))

            #if sum(hist[100:105]) > 0:
            if sum(hist[1000:1050]) > 0:

                unit_response += 1

            unit_trial_count.append(hist)

            if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.099)).sum() > 0:
            #if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.01)).sum() > 0:


                latency = min(i for i in spike_times[trial] if i > 1)

                latency = latency - 1

            else: 

                latency = float('NaN')

            unit_latency.append(latency)

        unit_latency = np.array(unit_latency)[(np.array(unit_latency) < 0.025) & (np.array(unit_latency) > 0.006)].tolist() 

        w1_trial_counts.append(np.nanmean(unit_trial_count, axis = 0))
        w1_resp_perc.append(unit_response)

        if np.array(unit_latency).sum() > 0: 

            w1_latency.append(np.nanmean(unit_latency,0))

        else: 

            w1_latency.append(float('Nan'))

    w2_trial_counts = []
    w2_resp_perc = []
    w2_latency = []

    for neuron in range(len(whisk_2)):

        spike_times = whisk_2[neuron]

        unit_trial_count = []

        unit_response = 0

        unit_latency = []

        for trial in range(len(whisk_2[neuron])):

            #hist, bins = np.histogram(spike_times[trial], 300, range = (0,3))
            hist, bins = np.histogram(spike_times[trial], 3000, range = (0,3))

            #if sum(hist[100:105]) > 0:
            if sum(hist[1000:1050]) > 0:

                unit_response += 1

            unit_trial_count.append(hist)

            if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.099)).sum() > 0:
            #if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.01)).sum() > 0:

                latency = min(i for i in spike_times[trial] if i > 1)

                latency = latency - 1

            else: 

                latency = float('NaN')

            unit_latency.append(latency)

        unit_latency = np.array(unit_latency)[(np.array(unit_latency) < 0.025) & (np.array(unit_latency) > 0.006)].tolist() 

        w2_trial_counts.append(np.nanmean(unit_trial_count, axis = 0))
        w2_resp_perc.append(unit_response)
        
        if np.array(unit_latency).sum() > 0: 

            w2_latency.append(np.nanmean(unit_latency,0))

        else: 

            w2_latency.append(float('Nan'))

    w1_bin_responses = [np.sum(resp[1000:1099]) for resp in w1_trial_counts]
    #w1_bin_responses = [np.sum(resp[100:105]) for resp in w1_trial_counts]
    w1_spont_responses = [np.sum(resp[800:899]) for resp in w1_trial_counts]
    #w1_spont_responses = [np.sum(resp[90:95]) for resp in w1_trial_counts]

    big_spont_responses = [np.sum(resp[500:600]) for resp in w1_trial_counts]
    w1_bin_responses = (np.array(w1_bin_responses) - np.array(w1_spont_responses)).tolist()

    w1_avg_response = np.mean(w1_bin_responses)

    w2_bin_responses = [np.sum(resp[1000:1099]) for resp in w2_trial_counts]
    #w2_bin_responses = [np.sum(resp[100:105]) for resp in w2_trial_counts]
    w2_spont_responses = [np.sum(resp[800:899]) for resp in w2_trial_counts]
    #w2_spont_responses = [np.sum(resp[90:95]) for resp in w2_trial_counts]
    w2_bin_responses = (np.array(w2_bin_responses) - np.array(w2_spont_responses)).tolist()

    w2_avg_response = np.mean(w2_bin_responses)

    if w1_avg_response > w2_avg_response:

    #if np.nanmean(np.array(w1_latency)) < np.nanmean(np.array(w2_latency)):

        print('PW = W1')

        pw_trial_counts = w1_trial_counts
        pw_resp_perc = w1_resp_perc
        pw_latency = w1_latency
        pw_bin_responses = w1_bin_responses

        aw_trial_counts = w2_trial_counts
        aw_resp_perc = w2_resp_perc
        aw_latency = w2_latency
        aw_bin_responses = w2_bin_responses

        pw_ID = 1

    if w1_avg_response < w2_avg_response:

    #if np.nanmean(np.array(w1_latency)) > np.nanmean(np.array(w2_latency)):

        print('PW = W2')

        pw_trial_counts = w2_trial_counts
        pw_resp_perc = w2_resp_perc
        pw_latency = w2_latency
        pw_bin_responses = w2_bin_responses

        aw_trial_counts = w1_trial_counts
        aw_resp_perc = w1_resp_perc
        aw_latency = w1_latency
        aw_bin_responses = w1_bin_responses

        pw_ID = 2

    return pw_ID, pw_trial_counts, pw_resp_perc, pw_latency, pw_bin_responses, aw_trial_counts, aw_resp_perc, aw_latency, aw_bin_responses, w1_avg_response, w2_avg_response, big_spont_responses


def dual_whisk_quad_analysis(pw_ID, whisk_1, whisk_2, w1_avg_response, w2_avg_response):

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

            if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.099)).sum() > 0:

                latency = min(i for i in spike_times[trial] if i > 1)

                latency = latency - 1

            else: 

                latency = float('NaN')

            unit_latency.append(latency)

        unit_latency = np.array(unit_latency)[(np.array(unit_latency) < 0.025) & (np.array(unit_latency) > 0.006)].tolist() 

        w1_trial_counts.append(np.nanmean(unit_trial_count, axis = 0))
        w1_resp_perc.append(unit_response)
        
        if np.array(unit_latency).sum() > 0: 

            w1_latency.append(np.nanmean(unit_latency,0))

        else: 

            w1_latency.append(float('Nan'))

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

            if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.099)).sum() > 0:

                latency = min(i for i in spike_times[trial] if i > 1)

                latency = latency - 1

            else: 

                latency = float('NaN')

            unit_latency.append(latency)

        unit_latency = np.array(unit_latency)[(np.array(unit_latency) < 0.025) & (np.array(unit_latency) > 0.006)].tolist() 

        w2_trial_counts.append(np.nanmean(unit_trial_count, axis = 0))
        w2_resp_perc.append(unit_response)
        
        if np.array(unit_latency).sum() > 0: 

            w2_latency.append(np.nanmean(unit_latency,0))

        else: 

            w2_latency.append(float('Nan'))
            

    w1_resp_1 = [np.sum(resp[1000:1025]) for resp in w1_trial_counts]
    w1_resp_2 = [np.sum(resp[1100:1125]) for resp in w1_trial_counts]
    w1_resp_3 = [np.sum(resp[1200:1225]) for resp in w1_trial_counts]
    w1_resp_4 = [np.sum(resp[1300:1325]) for resp in w1_trial_counts]
    
    w1_spont_responses = [np.sum(resp[800:825]) for resp in w1_trial_counts]
    
    w1_resp_1 = np.array(w1_resp_1) - np.array(w1_spont_responses).tolist()
    w1_resp_2 = np.array(w1_resp_2) - np.array(w1_spont_responses).tolist()
    w1_resp_3 = np.array(w1_resp_3) - np.array(w1_spont_responses).tolist()
    w1_resp_4 = np.array(w1_resp_4) - np.array(w1_spont_responses).tolist()
    
    w1_resp_1 = np.where(w1_resp_1<0, 0.001, w1_resp_1)
    w1_resp_2 = np.where(w1_resp_2<0, 0.001, w1_resp_2)
    w1_resp_3 = np.where(w1_resp_3<0, 0.001, w1_resp_3)
    w1_resp_4 = np.where(w1_resp_4<0, 0.001, w1_resp_4)
    
    w2_resp_1 = [np.sum(resp[1000:1025]) for resp in w2_trial_counts]
    w2_resp_2 = [np.sum(resp[1100:1125]) for resp in w2_trial_counts]
    w2_resp_3 = [np.sum(resp[1200:1225]) for resp in w2_trial_counts]
    w2_resp_4 = [np.sum(resp[1300:1325]) for resp in w2_trial_counts]
    
    w2_spont_responses = [np.sum(resp[800:825]) for resp in w2_trial_counts]
    
    w2_resp_1 = np.array(w2_resp_1) - np.array(w2_spont_responses).tolist()
    w2_resp_2 = np.array(w2_resp_2) - np.array(w2_spont_responses).tolist()
    w2_resp_3 = np.array(w2_resp_3) - np.array(w2_spont_responses).tolist()
    w2_resp_4 = np.array(w2_resp_4) - np.array(w2_spont_responses).tolist()
    
    w2_resp_1 = np.where(w2_resp_1<0, 0.001, w2_resp_1)
    w2_resp_2 = np.where(w2_resp_2<0, 0.001, w2_resp_2)
    w2_resp_3 = np.where(w2_resp_3<0, 0.001, w2_resp_3)
    w2_resp_4 = np.where(w2_resp_4<0, 0.001, w2_resp_4)

    #if w1_avg_response > w2_avg_response:

    #if np.nanmean(np.array(w1_latency)) < np.nanmean(np.array(w2_latency)):

    if pw_ID == 1:

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

        pw_1_latency = w1_latency
        aw_1_latency = w2_latency

    #    pw_ID = 1

   # if w1_avg_response < w2_avg_response:

    #if np.nanmean(np.array(w1_latency)) > np.nanmean(np.array(w2_latency)):

    if pw_ID == 2:

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

        pw_1_latency = w2_latency
        aw_1_latency = w1_latency

     #   pw_ID = 2
    
    pw_ratio_2_1 = (pw_quad_2/pw_quad_1)*100
    pw_ratio_4_1 = (pw_quad_4/pw_quad_1)*100
    
    aw_ratio_2_1 = (aw_quad_2/aw_quad_1)*100
    aw_ratio_4_1 = (aw_quad_4/aw_quad_1)*100

    return pw_quad_trial_counts, aw_quad_trial_counts, pw_1_latency, aw_1_latency, pw_quad_1, pw_quad_2, pw_quad_3, pw_quad_4, aw_quad_1, aw_quad_2, aw_quad_3, aw_quad_4, pw_ratio_2_1, pw_ratio_4_1, aw_ratio_2_1, aw_ratio_4_1 


def set_data_measure(df, opto_rs, non_opto_rs, measure, avg_type):

    import numpy as np
    import scipy.stats as st

    avg_opto = []
    avg_non_opto = []

    for dates in np.unique(df['date']):

        opto_date_mask = ((df['date'] == dates) & (opto_rs == True))
        non_opto_date_mask = ((df['date'] == dates) & (non_opto_rs == True))

        if ((opto_date_mask.sum() > 0) & (non_opto_date_mask.sum() > 0)):

            if avg_type == 'median':

                avg_opto.append(np.nanmedian(df[measure][opto_date_mask]))
                avg_non_opto.append(np.nanmedian(df[measure][non_opto_date_mask]))
            
            else:

                # avg_opto.append(np.nanmean(df[measure][opto_date_mask]))
                # avg_non_opto.append(np.nanmean(df[measure][non_opto_date_mask]))

                avg_opto.append(np.mean(df[measure][opto_date_mask]))
                avg_non_opto.append(np.mean(df[measure][non_opto_date_mask]))

        else: 
            
            print(dates, opto_date_mask.sum(), non_opto_date_mask.sum(), 'fail')

    avg_opto = [x for x in avg_opto if str(x) != 'nan']
    avg_non_opto = [x for x in avg_non_opto if str(x) != 'nan']

    data = [avg_opto, avg_non_opto]
    
    return data 


def plot_unit_pairs(data, bin_size, title, ylabel):

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    import seaborn as sns

    data = [np.array(data[0])/bin_size, np.array(data[1])/bin_size]
    
    plt.figure(figsize = (2,3))

    palette = ('limegreen', 'r')
    x_labels = ['aIP', 'OP']

    ax = sns.stripplot(data = data, size = 7, linewidth = 2, jitter = 0, palette = palette, zorder = 0)

    plt.hlines(np.mean(data[0]), -.1, .1)
    plt.hlines(np.mean(data[1]), .9, 1.1)

    plt.vlines(0, np.mean(data[0])-st.sem(data[0]),
                    np.mean(data[0])+st.sem(data[0]))

    plt.vlines(1, np.mean(data[1])-st.sem(data[1]),
                    np.mean(data[1])+st.sem(data[1]))

    for points in range(len(data[0])):

        plt.plot((0, 1), (data[0][points], data[1][points]), color = 'grey', alpha = 0.1)

    ax.set_xticklabels(x_labels)

    plt.xlim(-.5, 1.5)
    
    plt.title(title)
    plt.ylabel(ylabel)

    norm_0 = st.shapiro(data[0])[1]
    norm_1 = st.shapiro(data[1])[1]

    if (norm_0 > 0.05) & (norm_1 > 0.05): 

        print('data normal')
        print(st.ttest_rel(data[0], data[1]))

    else: 

        print('data not normal')
        print(st.wilcoxon(data[0], data[1]))

    #plt.tight_layout()


def plot_aIP_pairs(data, bin_size, title, ylabel):

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    import seaborn as sns

    data = [np.array(data[0])/bin_size, np.array(data[1])/bin_size]
    
    plt.figure(figsize = (1.5,3))

    palette = ('limegreen', 'darkolivegreen')
    x_labels = ['PW', 'AW']

    ax = sns.stripplot(data = data, size = 6, linewidth = 0, jitter = 0, palette = palette, zorder = 0)

    plt.hlines(np.mean(data[0]), -.1, .1)
    plt.hlines(np.mean(data[1]), .9, 1.1)

    plt.vlines(0, np.mean(data[0])-st.sem(data[0]),
                    np.mean(data[0])+st.sem(data[0]))

    plt.vlines(1, np.mean(data[1])-st.sem(data[1]),
                    np.mean(data[1])+st.sem(data[1]))

    for points in range(len(data[0])):

        plt.plot((0, 1), (data[0][points], data[1][points]), color = 'grey', alpha = 0.1)

    ax.set_xticklabels(x_labels)

    plt.xlim(-.5, 1.5)
    
    plt.title(title)
    plt.ylabel(ylabel)

    norm_0 = st.shapiro(data[0])[1]
    norm_1 = st.shapiro(data[1])[1]

    if (norm_0 > 0.05) & (norm_1 > 0.05): 

        print('data normal')
        print(st.ttest_rel(data[0], data[1]))

    else: 

        print('data not normal')
        print(st.wilcoxon(data[0], data[1]))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #plt.tight_layout()

def plot_OP_pairs(data, bin_size, title, ylabel):

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    import seaborn as sns

    data = [np.array(data[0])/bin_size, np.array(data[1])/bin_size]
    
    plt.figure(figsize = (1.5,3))

    palette = ('r', 'rosybrown')
    x_labels = ['PW', 'AW']

    ax = sns.stripplot(data = data, size = 6, linewidth = 0, jitter = 0, palette = palette, zorder = 0)

    plt.hlines(np.mean(data[0]), -.1, .1)
    plt.hlines(np.mean(data[1]), .9, 1.1)

    plt.vlines(0, np.mean(data[0])-st.sem(data[0]),
                    np.mean(data[0])+st.sem(data[0]))

    plt.vlines(1, np.mean(data[1])-st.sem(data[1]),
                    np.mean(data[1])+st.sem(data[1]))

    for points in range(len(data[0])):

        plt.plot((0, 1), (data[0][points], data[1][points]), color = 'grey', alpha = 0.1)

    ax.set_xticklabels(x_labels)

    plt.xlim(-.5, 1.5)
    
    plt.title(title)
    plt.ylabel(ylabel)

    norm_0 = st.shapiro(data[0])[1]
    norm_1 = st.shapiro(data[1])[1]

    if (norm_0 > 0.05) & (norm_1 > 0.05): 

        print('data normal')
        print(st.ttest_rel(data[0], data[1]))

    else: 

        print('data not normal')
        print(st.wilcoxon(data[0], data[1]))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    #plt.tight_layout()

def plot_versus_pairs(data, bin_size, title, ylabel):

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    import seaborn as sns

    data = [np.array(data[0])/bin_size, np.array(data[1])/bin_size]
    
    plt.figure(figsize = (1.5,3))

    palette = ('limegreen', 'r')
    x_labels = ['aIP', 'OP']

    ax = sns.stripplot(data = data, size = 6, linewidth = 0, jitter = 0, palette = palette, zorder = 0)

    plt.hlines(np.mean(data[0]), -.1, .1)
    plt.hlines(np.mean(data[1]), .9, 1.1)

    plt.vlines(0, np.mean(data[0])-st.sem(data[0]),
                    np.mean(data[0])+st.sem(data[0]))

    plt.vlines(1, np.mean(data[1])-st.sem(data[1]),
                    np.mean(data[1])+st.sem(data[1]))


    ax.set_xticklabels(x_labels)

    plt.xlim(-.5, 1.5)
    
    plt.title(title)
    plt.ylabel(ylabel)

    norm_0 = st.shapiro(data[0])[1]
    norm_1 = st.shapiro(data[1])[1]

    if (norm_0 > 0.05) & (norm_1 > 0.05): 

        print('data normal')
        print(st.ttest_ind(data[0], data[1]))

    else: 

        print('data not normal')
        print(st.wilcoxon(data[0], data[1]))

    #plt.tight_layout()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def plot_versus_unmatched_pairs(data, bin_size, title, ylabel):

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    import seaborn as sns

    data = [np.array(data[0])/bin_size, np.array(data[1])/bin_size]
    
    plt.figure(figsize = (1.5,3))

    palette = ('limegreen', 'r')
    x_labels = ['aIP', 'OP']

    ax = sns.stripplot(data = data, size = 6, linewidth = 0, jitter = 0, palette = palette, zorder = 0)

    plt.hlines(np.mean(data[0]), -.1, .1)
    plt.hlines(np.mean(data[1]), .9, 1.1)

    plt.vlines(0, np.mean(data[0])-st.sem(data[0]),
                    np.mean(data[0])+st.sem(data[0]))

    plt.vlines(1, np.mean(data[1])-st.sem(data[1]),
                    np.mean(data[1])+st.sem(data[1]))


    ax.set_xticklabels(x_labels)

    plt.xlim(-.5, 1.5)
    
    plt.title(title)
    plt.ylabel(ylabel)

    norm_0 = st.shapiro(data[0])[1]
    norm_1 = st.shapiro(data[1])[1]

    if (norm_0 > 0.05) & (norm_1 > 0.05): 

        print('data normal')
        print(st.ttest_ind(data[0], data[1]))

    else: 

        print('data not normal')
        print(st.ranksums(data[0], data[1]))

    #plt.tight_layout()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def plot_chr2neg_pairs(data, bin_size, title, ylabel):

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    import seaborn as sns

    data = [np.array(data[0])/bin_size, np.array(data[1])/bin_size]
    
    plt.figure(figsize = (1.5,3))

    palette = ('grey', 'lightgrey')
    x_labels = ['PW', 'AW']

    ax = sns.stripplot(data = data, size = 6, linewidth = 0, jitter = 0, palette = palette, zorder = 0)

    plt.hlines(np.mean(data[0]), -.1, .1)
    plt.hlines(np.mean(data[1]), .9, 1.1)

    plt.vlines(0, np.mean(data[0])-st.sem(data[0]),
                    np.mean(data[0])+st.sem(data[0]))

    plt.vlines(1, np.mean(data[1])-st.sem(data[1]),
                    np.mean(data[1])+st.sem(data[1]))

    for points in range(len(data[0])):

        plt.plot((0, 1), (data[0][points], data[1][points]), color = 'grey', alpha = 0.1)

    ax.set_xticklabels(x_labels)

    plt.xlim(-.5, 1.5)
    
    plt.title(title)
    plt.ylabel(ylabel)

    norm_0 = st.shapiro(data[0])[1]
    norm_1 = st.shapiro(data[1])[1]

    if (norm_0 > 0.05) & (norm_1 > 0.05): 

        print('data normal')
        print(st.ttest_ind(data[0], data[1]))

    else: 

        print('data not normal')
        print(st.wilcoxon(data[0], data[1]))

    #plt.tight_layout()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def plot_chr2pos_pairs(data, bin_size, title, ylabel):

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    import seaborn as sns

    data = [np.array(data[0])/bin_size, np.array(data[1])/bin_size]
    
    plt.figure(figsize = (1.5,3))

    palette = ('dodgerblue', 'lightsteelblue')
    x_labels = ['PW', 'AW']

    ax = sns.stripplot(data = data, size = 6, linewidth = 0, jitter = 0, palette = palette, zorder = 0)

    plt.hlines(np.mean(data[0]), -.1, .1)
    plt.hlines(np.mean(data[1]), .9, 1.1)

    plt.vlines(0, np.mean(data[0])-st.sem(data[0]),
                    np.mean(data[0])+st.sem(data[0]))

    plt.vlines(1, np.mean(data[1])-st.sem(data[1]),
                    np.mean(data[1])+st.sem(data[1]))

    for points in range(len(data[0])):

        plt.plot((0, 1), (data[0][points], data[1][points]), color = 'grey', alpha = 0.1)

    ax.set_xticklabels(x_labels)

    plt.xlim(-.5, 1.5)
    
    plt.title(title)
    plt.ylabel(ylabel)

    norm_0 = st.shapiro(data[0])[1]
    norm_1 = st.shapiro(data[1])[1]

    if (norm_0 > 0.05) & (norm_1 > 0.05): 

        print('data normal')
        print(st.ttest_ind(data[0], data[1]))

    else: 

        print('data not normal')
        print(st.wilcoxon(data[0], data[1]))

    #plt.tight_layout()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def plot_cag_pairs(data, bin_size, title, ylabel):

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    import seaborn as sns

    data = [np.array(data[0])/bin_size, np.array(data[1])/bin_size]
    
    plt.figure(figsize = (1.5,3))

    palette = ('dodgerblue', 'grey')
    x_labels = ['ChR2+', 'ChR2-']

    ax = sns.stripplot(data = data, size = 6, linewidth = 0, jitter = 0, palette = palette, zorder = 0)

    plt.hlines(np.mean(data[0]), -.1, .1)
    plt.hlines(np.mean(data[1]), .9, 1.1)

    plt.vlines(0, np.mean(data[0])-st.sem(data[0]),
                    np.mean(data[0])+st.sem(data[0]))

    plt.vlines(1, np.mean(data[1])-st.sem(data[1]),
                    np.mean(data[1])+st.sem(data[1]))


    ax.set_xticklabels(x_labels)

    plt.xlim(-.5, 1.5)
    
    plt.title(title)
    plt.ylabel(ylabel)

    norm_0 = st.shapiro(data[0])[1]
    norm_1 = st.shapiro(data[1])[1]

    if (norm_0 > 0.05) & (norm_1 > 0.05): 

        print('data normal')
        print(st.ttest_ind(data[0], data[1]))

    else: 

        print('data not normal')
        print(st.ranksums(data[0], data[1]))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def frequency_analysis(freq_spikes_1, freq_spikes_2, w1_avg_response, w2_avg_response):

    import numpy as np

    w1_trial_counts = []

    w1_freq_resps = []

    w1_freq_latency = []

    stim = [4,8,12,16,20]

    j = 0

    for frequency in freq_spikes_1:
        
        freq_trial_counts = []
        
        freq_resps = []

        freq_latency = []
        
        for unit in range(len(frequency)):
            
            spike_times = frequency[unit]
            
            unit_trial_counts = []
            
            unit_resps = []

            unit_latency = []
            
            for trial in range(len(spike_times)):
                
                hist, bins = np.histogram(spike_times[trial], bins = 5000, range = (0,5))
                                        
                unit_trial_counts.append(hist)                      
        
                stim_resp = []

                stim_latency = []
        
                stims = stim[j]*3
            
                start = 1000
                
                for whisk in range(stims):
                
                    i = int(1000/stim[j])
                
                    window_open = start+(whisk*i)
                    
                    window_close = start+(whisk*i)+40
        
                    spont = sum(hist[50:90])       

                    resp = sum(hist[window_open:window_close]) - spont

                    if np.argwhere((spike_times[trial] > (window_open/1000)) & (spike_times[trial] < (window_close/1000))).sum() > 0:

                        latency = min(i for i in spike_times[trial] if i > window_open/1000)
                        
                        latency = latency-window_open/1000

                    else:

                        latency = float('NaN')

                    stim_resp.append(resp)

                    stim_latency.append(latency)

                stim_resp = [0 if i < 0 else i for i in stim_resp]
                
                unit_resps.append(stim_resp)

                unit_latency.append(stim_latency)
            
            freq_resps.append(np.nanmean(unit_resps,0))
            
            freq_trial_counts.append(np.nanmean(unit_trial_counts,0))

            freq_latency.append(np.nanmean(unit_latency,0))
            
        j += 1
            
        w1_freq_resps.append(freq_resps)
        
        w1_trial_counts.append(freq_trial_counts)

        w1_freq_latency.append(freq_latency)
        
        
        
        
    w2_trial_counts = []

    w2_freq_resps = []

    w2_freq_latency = []

    stim = [4,8,12,16,20]

    j = 0

    for frequency in freq_spikes_2:
        
        freq_trial_counts = []
        
        freq_resps = []

        freq_latency = []
        
        for unit in range(len(frequency)):
            
            spike_times = frequency[unit]
            
            unit_trial_counts = []
            
            unit_resps = []

            unit_latency = []
            
            for trial in range(len(spike_times)):
                
                hist, bins = np.histogram(spike_times[trial], bins = 5000, range = (0,5))
                                        
                unit_trial_counts.append(hist)                      
        
                stim_resp = []

                stim_latency = []
        
                stims = stim[j]*3
            
                start = 1000
        
                for whisk in range(stims):
                
                    i = int(1000/stim[j])
                
                    window_open = start+(whisk*i)
                    
                    window_close = start+(whisk*i)+40                  
        
                    spont = sum(hist[50:90])       

                    resp = sum(hist[window_open:window_close]) - spont

                    if np.argwhere((spike_times[trial] > (window_open/1000)) & (spike_times[trial] < (window_close/1000))).sum() > 0:

                        latency = min(i for i in spike_times[trial] if i > window_open/1000)
                        
                        latency = latency-window_open/1000
                        
                    else:

                        latency = float('NaN')
                        
                    stim_resp.append(resp)

                    stim_latency.append(latency)
                    

                stim_resp = [0 if i < 0 else i for i in stim_resp]
                
            
            
                unit_resps.append(stim_resp)

                unit_latency.append(stim_latency)
                
            freq_resps.append(np.nanmean(unit_resps,0))
            
            freq_trial_counts.append(np.nanmean(unit_trial_counts,0))

            freq_latency.append(np.nanmean(unit_latency,0))
            
        j += 1
            
        w2_freq_resps.append(freq_resps)
        
        w2_trial_counts.append(freq_trial_counts)

        w2_freq_latency.append(freq_latency)
        


    if w1_avg_response > w2_avg_response:
            
            pw_freq_counts = w1_trial_counts
            aw_freq_counts = w2_trial_counts
            
            pw_freq_resps = w1_freq_resps
            aw_freq_resps = w2_freq_resps

            pw_freq_latency = w1_freq_latency
            aw_freq_latency = w2_freq_latency
            
    else:
            
            pw_freq_counts = w2_trial_counts
            aw_freq_counts = w1_trial_counts
            
            pw_freq_resps = w2_freq_resps
            aw_freq_resps = w1_freq_resps

            pw_freq_latency = w2_freq_latency
            aw_freq_latency = w1_freq_latency
                
    return pw_freq_counts, aw_freq_counts, pw_freq_resps, aw_freq_resps, pw_freq_latency, aw_freq_latency





def dual_whisk_single_analysis_10ms(whisk_1, whisk_2):

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

            hist, bins = np.histogram(spike_times[trial], 300, range = (0,3))
            #hist, bins = np.histogram(spike_times[trial], 3000, range = (0,3))

            #if sum(hist[100:105]) > 0:
            if sum(hist[1000:1050]) > 0:

                unit_response += 1

            unit_trial_count.append(hist)

            #if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.099)).sum() > 0:
            if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.01)).sum() > 0:


                latency = min(i for i in spike_times[trial] if i > 1)

                latency = latency - 1

            else: 

                latency = float('NaN')

            unit_latency.append(latency)

        unit_latency = np.array(unit_latency)[(np.array(unit_latency) < 0.02) & (np.array(unit_latency) > 0.005)].tolist() 

        w1_trial_counts.append(np.nanmean(unit_trial_count, axis = 0))
        w1_resp_perc.append(unit_response)

        if np.array(unit_latency).sum() > 0: 

            w1_latency.append(min(unit_latency))

        else: 

            w1_latency.append(float('Nan'))

    w2_trial_counts = []
    w2_resp_perc = []
    w2_latency = []

    for neuron in range(len(whisk_2)):

        spike_times = whisk_2[neuron]

        unit_trial_count = []

        unit_response = 0

        unit_latency = []

        for trial in range(len(whisk_2[neuron])):

            hist, bins = np.histogram(spike_times[trial], 300, range = (0,3))
            #hist, bins = np.histogram(spike_times[trial], 3000, range = (0,3))

            #if sum(hist[100:105]) > 0:
            if sum(hist[1000:1050]) > 0:

                unit_response += 1

            unit_trial_count.append(hist)

            #if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.099)).sum() > 0:
            if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.01)).sum() > 0:

                latency = min(i for i in spike_times[trial] if i > 1)

                latency = latency - 1

            else: 

                latency = float('NaN')

            unit_latency.append(latency)

        unit_latency = np.array(unit_latency)[(np.array(unit_latency) < 0.02) & (np.array(unit_latency) > 0.005)].tolist() 

        w2_trial_counts.append(np.nanmean(unit_trial_count, axis = 0))
        w2_resp_perc.append(unit_response)
        
        if np.array(unit_latency).sum() > 0: 

            w2_latency.append(min(unit_latency))

        else: 

            w2_latency.append(float('Nan'))

    #w1_bin_responses = [np.sum(resp[1000:1099]) for resp in w1_trial_counts]
    w1_bin_responses = [np.sum(resp[100:105]) for resp in w1_trial_counts]
    #w1_spont_responses = [np.sum(resp[800:899]) for resp in w1_trial_counts]
    w1_spont_responses = [np.sum(resp[90:95]) for resp in w1_trial_counts]

    big_spont_responses = [np.sum(resp[500:600]) for resp in w1_trial_counts]
    w1_bin_responses = (np.array(w1_bin_responses) - np.array(w1_spont_responses)).tolist()

    w1_avg_response = np.mean(w1_bin_responses)

    #w2_bin_responses = [np.sum(resp[1000:1099]) for resp in w2_trial_counts]
    w2_bin_responses = [np.sum(resp[100:105]) for resp in w2_trial_counts]
    #w2_spont_responses = [np.sum(resp[800:899]) for resp in w2_trial_counts]
    w2_spont_responses = [np.sum(resp[90:95]) for resp in w2_trial_counts]
    w2_bin_responses = (np.array(w2_bin_responses) - np.array(w2_spont_responses)).tolist()

    w2_avg_response = np.mean(w2_bin_responses)

    if w1_avg_response > w2_avg_response:

    #if np.nanmean(np.array(w1_latency)) < np.nanmean(np.array(w2_latency)):

        print('PW = W1')

        pw_trial_counts = w1_trial_counts
        pw_resp_perc = w1_resp_perc
        pw_latency = w1_latency
        pw_bin_responses = w1_bin_responses

        aw_trial_counts = w2_trial_counts
        aw_resp_perc = w2_resp_perc
        aw_latency = w2_latency
        aw_bin_responses = w2_bin_responses

        pw_ID = 1

    if w1_avg_response < w2_avg_response:

    #if np.nanmean(np.array(w1_latency)) > np.nanmean(np.array(w2_latency)):

        print('PW = W2')

        pw_trial_counts = w2_trial_counts
        pw_resp_perc = w2_resp_perc
        pw_latency = w2_latency
        pw_bin_responses = w2_bin_responses

        aw_trial_counts = w1_trial_counts
        aw_resp_perc = w1_resp_perc
        aw_latency = w1_latency
        aw_bin_responses = w1_bin_responses

        pw_ID = 2

    return pw_ID, pw_trial_counts, pw_resp_perc, pw_latency, pw_bin_responses, aw_trial_counts, aw_resp_perc, aw_latency, aw_bin_responses, w1_avg_response, w2_avg_response, big_spont_responses


def dual_whisk_quad_analysis_10ms(pw_ID, whisk_1, whisk_2, w1_avg_response, w2_avg_response):

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

            hist, bins = np.histogram(spike_times[trial], 300, range = (0,3))

            if sum(hist[1000:1050]) > 0:

                unit_response += 1

            unit_trial_count.append(hist)

            if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.01)).sum() > 0:

                latency = min(i for i in spike_times[trial] if i > 1)

                latency = latency - 1

            else: 

                latency = float('NaN')

            unit_latency.append(latency)

        unit_latency = np.array(unit_latency)[(np.array(unit_latency) < 0.02) & (np.array(unit_latency) > 0.005)].tolist() 

        w1_trial_counts.append(np.nanmean(unit_trial_count, axis = 0))
        w1_resp_perc.append(unit_response)
        
        if np.array(unit_latency).sum() > 0: 

            w1_latency.append(min(unit_latency))

        else: 

            w1_latency.append(float('Nan'))

    w2_trial_counts = []
    w2_resp_perc = []
    w2_latency = []

    for neuron in range(len(whisk_2)):

        spike_times = whisk_2[neuron]

        unit_trial_count = []

        unit_response = 0

        unit_latency = []

        for trial in range(len(whisk_2[neuron])):

            hist, bins = np.histogram(spike_times[trial], 300, range = (0,3))

            if sum(hist[1000:1050]) > 0:

                unit_response += 1

            unit_trial_count.append(hist)

            if np.argwhere((spike_times[trial] > 1) & (spike_times[trial] < 1.01)).sum() > 0:

                latency = min(i for i in spike_times[trial] if i > 1)

                latency = latency - 1

            else: 

                latency = float('NaN')

            unit_latency.append(latency)

        unit_latency = np.array(unit_latency)[(np.array(unit_latency) < 0.02) & (np.array(unit_latency) > 0.005)].tolist() 

        w2_trial_counts.append(np.nanmean(unit_trial_count, axis = 0))
        w2_resp_perc.append(unit_response)
        
        if np.array(unit_latency).sum() > 0: 

            w2_latency.append(min(unit_latency))

        else: 

            w2_latency.append(float('Nan'))
            

    w1_resp_1 = [np.sum(resp[100:1030]) for resp in w1_trial_counts]
    w1_resp_2 = [np.sum(resp[110:1130]) for resp in w1_trial_counts]
    w1_resp_3 = [np.sum(resp[120:1230]) for resp in w1_trial_counts]
    w1_resp_4 = [np.sum(resp[130:1330]) for resp in w1_trial_counts]
    
    w1_spont_responses = [np.sum(resp[800:830]) for resp in w1_trial_counts]
    
    w1_resp_1 = np.array(w1_resp_1) - np.array(w1_spont_responses).tolist()
    w1_resp_2 = np.array(w1_resp_2) - np.array(w1_spont_responses).tolist()
    w1_resp_3 = np.array(w1_resp_3) - np.array(w1_spont_responses).tolist()
    w1_resp_4 = np.array(w1_resp_4) - np.array(w1_spont_responses).tolist()
    
    w1_resp_1 = np.where(w1_resp_1<0, 0.001, w1_resp_1)
    w1_resp_2 = np.where(w1_resp_2<0, 0.001, w1_resp_2)
    w1_resp_3 = np.where(w1_resp_3<0, 0.001, w1_resp_3)
    w1_resp_4 = np.where(w1_resp_4<0, 0.001, w1_resp_4)
    
    w2_resp_1 = [np.sum(resp[100:1030]) for resp in w2_trial_counts]
    w2_resp_2 = [np.sum(resp[110:1130]) for resp in w2_trial_counts]
    w2_resp_3 = [np.sum(resp[120:1230]) for resp in w2_trial_counts]
    w2_resp_4 = [np.sum(resp[130:1330]) for resp in w2_trial_counts]
    
    w2_spont_responses = [np.sum(resp[800:830]) for resp in w2_trial_counts]
    
    w2_resp_1 = np.array(w2_resp_1) - np.array(w2_spont_responses).tolist()
    w2_resp_2 = np.array(w2_resp_2) - np.array(w2_spont_responses).tolist()
    w2_resp_3 = np.array(w2_resp_3) - np.array(w2_spont_responses).tolist()
    w2_resp_4 = np.array(w2_resp_4) - np.array(w2_spont_responses).tolist()
    
    w2_resp_1 = np.where(w2_resp_1<0, 0.001, w2_resp_1)
    w2_resp_2 = np.where(w2_resp_2<0, 0.001, w2_resp_2)
    w2_resp_3 = np.where(w2_resp_3<0, 0.001, w2_resp_3)
    w2_resp_4 = np.where(w2_resp_4<0, 0.001, w2_resp_4)

    #if w1_avg_response > w2_avg_response:

    #if np.nanmean(np.array(w1_latency)) < np.nanmean(np.array(w2_latency)):

    if pw_ID == 1:

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

        pw_1_latency = w1_latency
        aw_1_latency = w2_latency

    #    pw_ID = 1

   # if w1_avg_response < w2_avg_response:

    #if np.nanmean(np.array(w1_latency)) > np.nanmean(np.array(w2_latency)):

    if pw_ID == 2:

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

        pw_1_latency = w2_latency
        aw_1_latency = w1_latency

     #   pw_ID = 2
    
    pw_ratio_2_1 = (pw_quad_2/pw_quad_1)*100
    pw_ratio_4_1 = (pw_quad_4/pw_quad_1)*100
    
    aw_ratio_2_1 = (aw_quad_2/aw_quad_1)*100
    aw_ratio_4_1 = (aw_quad_4/aw_quad_1)*100

    return pw_quad_trial_counts, aw_quad_trial_counts, pw_1_latency, aw_1_latency, pw_quad_1, pw_quad_2, pw_quad_3, pw_quad_4, aw_quad_1, aw_quad_2, aw_quad_3, aw_quad_4, pw_ratio_2_1, pw_ratio_4_1, aw_ratio_2_1, aw_ratio_4_1 


def intrinsic_timescale(data, time):
    
    import numpy as np
    

    w_autocorr = np.zeros((len(data), 25))

    w_tau = np.zeros((len(data)))

    w_fit = np.zeros((len(data), 25))

    for counter, neuron in enumerate(data):
    
        hist = [np.histogram(trial, 300, range = (0,3))[0] for trial in neuron]
    
        corr = [autocorr(trial[100:125]) for trial in hist]

        w_autocorr[counter] = np.nanmean(corr,0)

    
    for counter, neuron in enumerate(w_autocorr):
    
        if neuron.sum() > 0:
        
            w_tau[counter] = compute_intrinsic_tau(time, neuron)[0]
            w_fit[counter] = compute_intrinsic_tau(time, neuron)[1]
        
        else: 
        
            w_tau[counter] = float('NaN')
            w_fit[counter] = float('NaN')
            
    return w_autocorr, w_tau, w_fit

def autocorr(x):
    
    import numpy as np 
    import scipy.ndimage as nd 
    
    result = np.correlate(x, x, mode='full')
    result = nd.gaussian_filter(result,1)
    result = result / float(result.max())
    return result[result.size // 2:]

def exponential(x, a, b):
    import numpy as np
    
    return a*np.exp(b*x)

def compute_intrinsic_tau(x_data, y_data):
    
    from scipy.optimize import curve_fit
    import numpy as np
    
    y_data = autocorr(y_data)
    pars, cov = curve_fit(f=exponential, xdata=x_data, ydata=y_data, p0=[0, 0], bounds=(-np.inf, np.inf))
    exp = exponential(x_data, *pars)
    tau = -1/pars[1]

    return tau, exp