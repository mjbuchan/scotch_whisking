def plasticity_pre_post(data_pre, data_post):

    import numpy as np
    import scipy.stats as st
    

    pre_resp = []
    pre_trace = []

    for neuron in data_pre:

        hist = [np.histogram(trial, 3000, range = (0,3))[0] for trial in neuron]

        spont = np.array([np.sum(trial) for trial in hist])/12
        #spont = 0

        resp = [np.sum(trial[1000:1250]) for trial in hist]

        pre_resp.append((np.array(resp)-np.array(spont)).tolist())
        
        pre_trace.append(np.mean(hist,0))

    pre_resp = np.mean(pre_resp,0)
    
    pre_trace_sem = st.sem(pre_trace,0)
    pre_trace = np.mean(pre_trace,0)
    


    post_resp = []
    post_trace = []

    for neuron in data_post:

        hist = [np.histogram(trial, 3000, range = (0,3))[0] for trial in neuron]

        spont = np.array([np.sum(trial) for trial in hist])/12
        #spont = 0

        resp = [np.sum(trial[1000:1250]) for trial in hist]

        post_resp.append((np.array(resp)-np.array(spont)).tolist())
        
        post_trace.append(np.mean(hist,0))

    post_resp = np.mean(post_resp,0)

    mean_pre_resp = np.mean(pre_resp)
    std_pre_resp = np.std(pre_resp)
    
    mean_post_resp = np.mean(post_resp)

    pre_resp = (pre_resp-mean_pre_resp)/std_pre_resp
    post_resp = (post_resp-mean_pre_resp)/std_pre_resp
    
    post_trace_sem = st.sem(post_trace,0)
    post_trace = np.mean(post_trace,0)
    

    return pre_resp, post_resp, pre_trace, post_trace, mean_pre_resp, mean_post_resp, pre_trace_sem, post_trace_sem


def plasticity_preprocess(path):

    import numpy as np 
    import os 
    from scipy.io import loadmat
    from scotch_whisking import plasticity as pp
    import scipy.stats as st

    avg_pre = []
    avg_post = []

    avg_pre_mua = []
    avg_post_mua = []

    avg_pre_raw = []
    avg_post_raw = []

    avg_pre_trace = []
    avg_post_trace = []

    for date in os.listdir(path): 

        if os.path.isdir(os.path.join(path, date)):

            pre = loadmat(os.path.join(path,date,'pre/spikes.mat'))
            pre = pre['spikes']

            post = loadmat(os.path.join(path,date,'post/spikes.mat'))
            post = post['spikes']

            pre_mua = loadmat(os.path.join(path,date,'pre/mua_spikes.mat'))
            pre_mua = pre_mua['mua_spikes']

            post_mua = loadmat(os.path.join(path,date,'post/mua_spikes.mat'))
            post_mua = post_mua['mua_spikes']

            pre_resp, post_resp, pre_trace, post_trace, mean_pre_resp, mean_post_resp, pre_trace_sem, post_trace_sem = pp.plasticity_pre_post(pre_mua[0:8], post_mua[0:8])

            avg_pre.append(pre_resp)
            avg_post.append(post_resp)

            avg_pre_mua.append(np.nanmean(pre_mua,0))
            avg_post_mua.append(np.nanmean(post_mua,0))

            avg_pre_raw.append(mean_pre_resp)
            avg_post_raw.append(mean_post_resp)

            avg_pre_trace.append(pre_trace)
            avg_post_trace.append(post_trace)
            
    data = [np.array(avg_pre_raw)/0.25, np.array(avg_post_raw)/0.25]
    data_delta = (np.array(avg_post_raw)/np.array(avg_pre_raw))*100

    res_pre = np.mean(np.array(np.nanmean(avg_pre,0)).reshape(-1, 4), axis=1)
    std_res_pre = st.sem(np.array(np.nanmean(avg_pre,0)).reshape(-1, 4), axis=1)

    res_post = np.mean(np.array(np.nanmean(avg_post,0)).reshape(-1, 4), axis=1)
    std_res_post = st.sem(np.array(np.nanmean(avg_post,0)).reshape(-1, 4), axis=1)

    return avg_pre, avg_post, avg_pre_mua, avg_post_mua, avg_pre_trace, avg_post_trace, data, data_delta, res_pre, res_post, std_res_pre, std_res_post


def plot_pre_post_trace(avg_pre_trace, avg_post_trace, condition):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    import scipy.ndimage as nd

    if condition == 'chr2':
        
        palette = ['grey', 'limegreen']
        
    if condition == 'lhx2':
        
        palette = ['grey', 'rebeccapurple']

    pre_trace = np.mean(avg_pre_trace,0)
    post_trace = np.mean(avg_post_trace,0)

    pre_trace_sem = st.sem(avg_pre_trace,0)
    post_trace_sem = st.sem(avg_post_trace,0)

    plt.figure(figsize = (1.5,1.5))

    plotting_pre = nd.gaussian_filter(pre_trace-np.mean(avg_pre_trace),1)[950:1250]/0.001
    plotting_post = nd.gaussian_filter(post_trace-np.mean(post_trace),1)[950:1250]/0.001

    plotting_pre_sem = nd.gaussian_filter(np.array(pre_trace_sem),1)[950:1250]/0.001
    plotting_post_sem = nd.gaussian_filter(np.array(post_trace_sem),1)[950:1250]/0.001

    plt.plot(np.arange(-50, 250, 1), plotting_pre, color = palette[0], linewidth = 2)
    plt.plot(np.arange(-50, 250, 1), plotting_post, color = palette[1], linewidth = 2)

    plt.fill_between(np.arange(-50, 250, 1), plotting_pre - 2*plotting_pre_sem, plotting_pre + 2*plotting_pre_sem, color = palette[0], alpha = 0.2)
    plt.fill_between(np.arange(-50, 250, 1), plotting_post - 2*plotting_post_sem, plotting_post + 2*plotting_post_sem, color = palette[1], alpha = 0.2)

    plt.xlabel('Time (ms)')
    plt.ylabel('Spike rate (Hz)')

    plt.text(130,900, 'Pre', color = palette[0])
    plt.text(130,700, 'Post', color = palette[1])

    plt.xlim(-50, 250)
    plt.ylim(0,1200)

def plot_pre_post_resp(res_pre, res_post, std_res_pre, std_res_post, condition):
    
    import numpy as np
    import matplotlib.pyplot as plt

    if condition == 'chr2':
        
        palette = ['grey', 'limegreen']
        
    if condition == 'lhx2':
        
        palette = ['grey', 'rebeccapurple']
        

    plt.figure(figsize = (3,3))

    ax = plt.scatter(np.arange(-1000/60,0,(1000/60)/25),res_pre, color = palette[1], s = 12)
    ax = plt.scatter(np.arange(60/60,1060/60,(1000/60)/25),res_post, color = palette[1], s = 12)

    ax = plt.fill_between(np.arange(-1000/60,0,(1000/60)/25), res_pre - 2*std_res_pre, res_pre+2*std_res_pre, alpha = 0.2, color = palette[1])
    ax = plt.fill_between(np.arange(60/60,1060/60,(1000/60)/25), res_post - 2*std_res_post, res_post+2*std_res_post, alpha = 0.2, color = palette[1])

    plt.xlim(-17,17)
    plt.ylim(-2,6)

    plt.hlines(0, -1000/60, 1000/60, ls = ':', color = 'black')
    plt.vlines(0.2, -2, 6, color = 'black', linewidth = 3, alpha = 0.2)

    plt.text(-7, -1.5, 'RWS', color = 'black')

    plt.xlabel('Time (min)')
    plt.ylabel('Normalised spike rate')

    if condition == 'lhx2':
        plt.text(-10,4, 'WT', color = 'limegreen')
        plt.text(-10,3,'Lhx2+', color = 'rebeccapurple')

def plot_pre_post_paired(data, condition):  

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    import seaborn as sns

    
    plt.figure(figsize = (1.5,1.5))

    if condition == 'chr2':
        
        palette = ['grey', 'limegreen']
        
    if condition == 'lhx2':
        
        palette = ['grey', 'rebeccapurple']

    ax = sns.stripplot(data = data, jitter=0,color = 'black', palette = palette, zorder = 1)
    ax = sns.barplot(data=data, facecolor = 'white', edgecolor = 'black', zorder = 0, ci = None)

    for points in range(len(data[0])):

        plt.plot((0, 1), (data[0][points], data[1][points]), color = 'black', linewidth = 0.2)

    ax.set_xticklabels(['pre', 'post'], rotation = 0)

    plt.xlim(-.75, 1.75)

    plt.ylim(0,300)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    print(st.ttest_rel(data[0], data[1]))

    plt.ylabel('Spike rate (Hz)')

    # plt.hlines(np.mean(data[0]), -.1,0.1, color = 'black')
    # plt.hlines(np.mean(data[1]), .9,1.1, color = 'black')

    plt.vlines(0, np.mean(data[0])-st.sem(data[0]), np.mean(data[0])+st.sem(data[0]), color = 'black')
    plt.vlines(1, np.mean(data[1])-st.sem(data[1]), np.mean(data[1])+st.sem(data[1]), color = 'black')

    if condition == 'chr2':

        plt.hlines(275, 0, 1, color = 'black')
        plt.text(0.325, 280, '**')


def plot_single_example(pre_resp, post_resp, condition):

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as st 

    plt.figure(figsize = (1.5,1.5))

    if condition == 'chr2':
    
        palette = ['grey', 'limegreen']
        
    if condition == 'lhx2':
        
        palette = ['grey', 'rebeccapurple']

    plt.scatter(np.arange(-1000/60,0,(1000/60)/100), pre_resp, label = 'pre', s =12, color = palette[1])
    plt.scatter(np.arange(60/60,1060/60,(1000/60)/100), post_resp, label = 'post', s = 12, color = palette[1])

    plt.xlabel('Time (min)')
    plt.ylabel('Normalised \nspike rate')
    
    plt.xlim(-18, 18)
    plt.ylim(-4, 6)
    
    plt.hlines(np.mean(pre_resp,0), -1000/60, 1000/60, ls = ':', color = 'black')
    plt.vlines(0.3, -50, 150, color = 'black', linewidth = 3, alpha = 0.2)

    plt.text(-15, 5, 'RWS', color = 'black')


def plot_single_traces(pre_trace, post_trace, condition):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.ndimage as nd
    
    if condition == 'chr2':
    
        palette = ['grey', 'limegreen']
        
    if condition == 'lhx2':
        
        palette = ['grey', 'rebeccapurple']
    
    plt.figure(figsize = (1.5,1.5))

    plt.plot(np.arange(-50,250,1), (nd.gaussian_filter(pre_trace/0.001,2)[950:1250]-np.mean(pre_trace,0)), color = palette[0], linewidth = 2)
    plt.plot(np.arange(-50,250,1), (nd.gaussian_filter(post_trace/0.001,2)[950:1250]-np.mean(post_trace,0)), color = palette[1], linewidth = 2)

    plt.xlabel('Time (ms)')
    plt.ylabel('Spike Rate (Hz)')

    plt.ylim(0,1200)

    plt.text(130, 900, 'Pre', color = palette[0])
    plt.text(130, 700, 'Post', color = palette[1])

def generate_single_example(date, condition):

    import numpy as np 
    import os
    import scipy.stats as st
    import scipy.ndimage as nd
    from scipy.io import loadmat
    
    
    gen_path = '/Users/matthewbuchan/Desktop/data_backup/new_plasticity/data/' 
    path = os.path.join(gen_path, condition)
    
    pre_mua = loadmat(os.path.join(path,date,'pre/mua_spikes.mat'))
    pre_mua = pre_mua['mua_spikes']

    post_mua = loadmat(os.path.join(path,date,'post/mua_spikes.mat'))
    post_mua = post_mua['mua_spikes']

    pre_resp = []
    pre_trace = []
    pre_opto = []

    for neuron in pre_mua[0:12]:

        hist = [np.histogram(trial, 3000, range = (0,3))[0] for trial in neuron]

        spont = np.array([np.sum(trial[0:3000]) for trial in hist])/12

        resp = [np.sum(trial[1000:1250]) for trial in hist]
        
        #opto_resp = [np.sum(trial[600:605]) for trial in hist]

        pre_resp.append((np.array(resp)-np.array(spont)).tolist())
        #pre_resp.append((np.array(resp).tolist())
        
        pre_trace.append(np.mean(hist,0))

    pre_resp = np.mean(pre_resp,0)
    
    pre_trace = np.mean(pre_trace,0)
    
  #  pre_opto.append((np.mean(opto_resp) -np.array(spont)).tolist())


    post_resp = []
    post_trace = []
    post_opto = []

    for neuron in post_mua[0:12]:

        hist = [np.histogram(trial, 3000, range = (0,3))[0] for trial in neuron]

        spont = np.array([np.sum(trial[0:3000]) for trial in hist])/12

        resp = [np.sum(trial[1000:1250]) for trial in hist]
        
       # opto_resp = [np.sum(trial[600:605]) for trial in hist]

        post_resp.append((np.array(resp)-np.array(spont)).tolist())
        #post_resp.append((np.array(resp).tolist())
        
        post_trace.append(np.mean(hist,0))
        
   # post_opto.append((np.mean(opto_resp) -np.array(spont)).tolist())

    post_resp = np.mean(post_resp,0)

    mean_pre_resp = np.mean(pre_resp)
    std_pre_resp = np.std(pre_resp)
    
  #  mean_pre_opto = np.mean(pre_opto)
  #  std_pre_opto = np.std(pre_opto)

    pre_resp = (pre_resp-mean_pre_resp)/std_pre_resp
    post_resp = (post_resp-mean_pre_resp)/std_pre_resp
    
    #pre_opto = (pre_opto - mean_pre_opto)/std_pre_opto
    #post_opto = (post_opto - mean_pre_opto)/std_pre_opto
    
    post_trace = np.mean(post_trace,0)

    return pre_resp, post_resp, pre_trace, post_trace#, pre_opto, post_opto

