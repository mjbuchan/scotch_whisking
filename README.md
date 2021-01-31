# scotch_whisking 

Code for analysis of in vivo electrophysiology data for Akerman Lab, Oxford. Written by a Scot abroad. Currently used for 32ch Neuronexus laminar probes in mouse cortex. Designed around Kilosort 1.0 output but can easily be tweaked to suit requirements.

## Requirements 

Python 3.7 

### Contents 

Functions grouped into pre-processing, analysis and storage files. Preprocessing functions include:
* Waveform analysis 
   * i.e. clustering and plotting of putative fast-spiking/regular-spiking units
* Cortical layer definition 
   * Uses current source density of LFP to identify current sources/sinks. This can be used to identify cortical layers. 
* Opto-tagging
   * Uses average spike latency and jitter to cluster putative channelrhodopsin-expressing neurons 
   
Analysis functions include:
* Entropy
  * Calculates Shannon entropy based on probability mass function of interspike intervals
* Coupling 
  * Calculates network coupling of sorted units with respect to either the LFP, multi-unit activity or summed unit activity of local neurons. Analogous to a sliding dot product of two vectors. (Inspired by https://www.nature.com/articles/nature14273) 
* Single and repeated whisker stimulation 
  * Calculates binned, cumulative, and peak spiking activity in response to the stimulation of one or several whiskers
* Pairwise phase consistency
  * Uses LFP and unit spike times to calculate a bias-free measure of phase coupling (Inspired by https://pubmed.ncbi.nlm.nih.gov/20114076/)
* Intrinsic timescale
  * Fits exponential to autocorrelation of sorted unit activity - can be used on spontaneous or evoked data
  
Storage functions include:
* Construction of dataframes grouped by experiment type from pre-processing and analysis output.
  * Outputs pickled Pandas dataframe of specified dates and experiment types. 

### Inputs 

Only required data are: 
* array: units x trials x spike_times (for each experiment type - ideally, spontaneous activity, optogenetic stimulation and sensory stimulation)
* array: unit_depths (len = units)
If you want to take advantage of some analysis methods, also require:
* array: channels x trials x spike_times (i.e. multiunit activity/LFP)
* array: unit_waveforms (len = units)

Currently this is acheived using Kilosort 1.0 Matlab output.

#### To do

* Need to finish de-bugging coupling functionality, help welcome 
* Carefully remove any comments resulting from a fragile state of mind (oops)
* Complete higher order processing functionality a lá https://www.nature.com/articles/s41586-020-03171-x#Sec1


