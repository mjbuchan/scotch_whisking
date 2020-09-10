def pmf_from_counts(counts):
  """Given counts, normalize by the total to estimate probabilities.
  
  Neuromatch academy - altered by Matt Buchan // Akerman Lab - Aug 2020
  """
  import numpy as np

  pmf = counts / np.sum(counts)
  return pmf


def plot_pmf(pmf,isi_range):
  """Plot the probability mass function.
  
  Neuromatch academy - altered by Matt Buchan // Akerman Lab - Aug 2020
  """
  import numpy as np
  import matplotlib.pyplot as plt

  ymax = max(0.2, 1.05 * np.max(pmf))
  pmf_ = np.insert(pmf, 0, pmf[0])
  plt.plot(bins, pmf_, drawstyle="steps")
  plt.fill_between(bins, pmf_, step="pre", alpha=0.4)
  plt.xlabel("Inter-spike interval (s)")
  plt.ylabel("Probability mass")
  plt.xlim(isi_range)
  plt.ylim([0, ymax])


def entropy(pmf):
  """Given a discrete distribution, return the Shannon entropy in bits.
  This is a measure of information in the distribution. For a totally
  deterministic distribution, where samples are always found in the same bin,
  then samples from the distribution give no more information and the entropy
  is 0.
  For now this assumes `pmf` arrives as a well-formed distribution (that is,
  `np.sum(pmf)==1` and `not np.any(pmf < 0)`)
  Args:
    pmf (np.ndarray): The probability mass function for a discrete distribution
      represented as an array of probabilities.
  Returns:
    h (number): The entropy of the distribution in `pmf`.
    

  Neuromatch academy - altered by Matt Buchan // Akerman Lab - Aug 2020
  """
  import numpy as np

  # reduce to non-zero entries to avoid an error from log2(0)
  pmf = pmf[pmf > 0]

  # implement the equation for Shannon entropy (in bits)
  h = -np.sum(pmf * np.log2(pmf))

  # return the absolute value (avoids getting a -0 result)
  return np.abs(h)

def calculate_unit_entropy(data):
    
    '''Takes neurons x trials x spike times and gives entropy for each unit
    
    
    Matt Buchan // Akerman Lab - Aug 2020
    '''

    import numpy as np

    unit_entropy = []

    isi_range = (0,0.25)
    n_bins = 50

    bins = np.linspace(*isi_range, n_bins + 1)

    for i in range(len(data)):

        unit = data[i,:,:]

        isi = []

        for j in range(len(unit)):

            isi.append(np.diff(unit[j]))

        isi = (np.array(isi)).flatten()
        isi = isi[~np.isnan(isi)]

        counts, _ = np.histogram(isi,bins)

        pmf = pmf_from_counts(counts)

        unit_entropy.append(entropy(pmf))
        
    return unit_entropy




def calculate_coupling(spikes, lfp, mua, method):


    '''
    Matt Buchan // Akerman Lab - Sept 2020
    '''

    
    import numpy as np
    import scipy.ndimage as nd

    lfp_step = 500
    mua_step = 50

    #Sorted units
    trial_counts = []

    if method == 'single_unit':

        #for each unit
        for i in range(len(spikes)):

            # for each trial
            trial_count_per_unit = []

            for j in range(len(spikes[i])):

                hist, bins = np.histogram(spikes[i][j], 1000, range = (0,10))

                trial_count_per_unit.append(hist)

            trial_counts.append(trial_count_per_unit)

        #MUA ATIVITY FROM SINGLE UNITS 

        mua_counts = np.sum(trial_counts, axis = 0)

        for i in range(len(mua_counts)):

            mua_counts[i] = nd.gaussian_filter1d(mua_counts[i], sigma = 1.2)

    else:

        #MUA ACTIVITY
        mua_trial_counts = []

        #for each channel
        for i in range(len(mua)):

            #for each trial    
            trial_count_per_channel = []

            for j in range(len(mua[i])):

               hist, bins = np.histogram(mua
                                         [i][j], 1000, range = (0,10))

               trial_count_per_channel.append(hist)

            mua_trial_counts.append(trial_count_per_channel)

        #SUM ACROSS CHANNELS 

        mua_counts = (np.sum(mua_trial_counts, axis = 0))*100

        #smooth mua with 12sd gaussian
        for i in range(len(mua_counts)):

            mua_counts[i] = nd.gaussian_filter1d(mua_counts[i], sigma = 1.2)


    #STLFP
    population_stlfp = []

    #for each unit
    for i in range(len(trial_counts)):

        unit_stlfp = []

        #for each trial histogram
        for j in range(len(trial_counts[i])):

            for z in range(len(trial_counts[i][j])):

                if trial_counts[i][j][z] == 1:

                    start = (z*10) - lfp_step 
                    stop = (z*10) + lfp_step

                    if (start > 0) & (stop < len(lfp[j])):

                        lfp_slice = lfp[j][start:stop]
                        unit_stlfp.append(lfp_slice)


        population_stlfp.append(np.nanmean(unit_stlfp, axis = 0))

    #STPR
    population_stpr = []
    stpr_1st_half = []
    stpr_2nd_half = []

    #for each unit
    for i in range(len(trial_counts)):

        unit_stpr = []

        #for each trial histogram
        for j in range(len(trial_counts[i])):

            for z in range(len(trial_counts[i][j])):

                if trial_counts[i][j][z] == 1:

                    start = (z) - mua_step 
                    stop = (z) + mua_step

                    if (start > 0) & (stop < len(mua_counts[j])):

                        mua_slice = mua_counts[j][start:stop]
                        unit_stpr.append(mua_slice)

        population_stpr.append(np.nanmean(unit_stpr, axis = 0))

        half = len(unit_stpr)//2 

        stpr_1st_half.append(np.mean(unit_stpr[:half], axis = 0))
        stpr_2nd_half.append(np.mean(unit_stpr[half:], axis = 0))

    population_stpr = (np.asarray(population_stpr))*10

    unit_mua_coupling = []

    for i in range(len(population_stpr)):

      if (population_stpr[i]).sum() > 1:

        unit_mua_coupling.append(population_stpr[i][50])

      else:

        unit_mua_coupling.append(float('NaN'))

    unit_lfp_coupling = []
    
    for i in range(len(population_stlfp)):

      if (population_stlfp[i]).sum() > 1:

        unit_lfp_coupling.append(population_stlfp[i][500])

      else:

        unit_lfp_coupling.append(float('NaN'))

    return population_stpr, population_stlfp, unit_mua_coupling, unit_lfp_coupling