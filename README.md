Note: still very much in beta. It "works for us", but is built to also work for others with minor modificaiton. There are still a few improvements to be made under the hood though. 

# DataMerge

Datamerge is a command-line tool intended for statistically sound merging and/or (re)binning of input datasets. It is a pretty flexible program that can be used for a few different things:

1. initial binning of 2D data into a 1D curve. 
2. rebinning a 1D dataset to use fewer points or using a different spacing arrangement between points (see details for faster execution below)
3. merging multiple datasets into a single, wide-range dataset. This is great for getting good datapoint statistics. 

It can probably also be run from a Jupyter notebook, but the main intention is to make this part of the standard data processing pipeline. 

It runs using settings read from a configuration file (in YAML format), see mergeConfig.yaml in default or mergeConfigExample.yaml in tests. 

If you would like to test this, the required dataset is available here: 
https://figshare.com/articles/dataset/Test_dataset_for_dataMerge/21655646

once unpacked in the correct directory ([your datamerge directory]/tests/), datamerge can be run using the command line from its directory: 

'''python
python main.py -f .\tests\data\20220925\autoproc\group_6\ -C .\tests\mergeConfigExample.yaml -o test.nxs
'''

## Input

Input files can be provided as a list on the command line, or as a path to a directory. In that case, the files are found using the "globKey" provided (if not default). Input files should be NeXus files processed using DAWN (reader is in the readersandwriters). 

## Input processing

Input files are read, their qmin and qmax determined, and optionally clipped to a range specified in the datamerge configuration file. These clipping settings can be set for none, some or all input files, either based on index (input files are sorted with smallest qmin first), or based on configuration setting ID. 

## Autoscaling

Data can be autoscaled, this helps when one dataset's vertical scaling is more reliable. Other datasets can be scaled to this. 

## Merging

Merging is done on a bin-by-bin basis. By default, the datapoints are weighted by their uncertainty: uncertain datapoints are weighted less heavily than accurate datapoints. This has been described in [our recent paper](https://iopscience.iop.org/article/10.1088/1748-0221/16/06/P06034). Additional uncertainty estimates in both I and Q are determined during the binning procedure, which can be used in subsequent analyses. 

The implementation includes statistical functions for averaging and uncertainty estimation written in cython, to allow faster rebinning. To employ this faster implementation there is a `binstats.pyx` script that needs to be compiled.This is however integrated into the setup.py script available, such that the only requirment to use it, is installed Cython (pip install Cython). Upon installation of a package `binstats.so` instance will be automatically created  (or .pyd on Windows) compartible with your operational system. 

In case you dont want to use the cythonized version, comment out the last line in the `setup.py` script and use functions for merging as is.

## Output

Datamerge outputs to a NeXus file, containing the results, the datamerge settings, and the input datasets. Output and input datasets can be plotted together in software such as DAWN without further adjustment. 

## Plotting

By default, plots are made that allow you to check the configuration of your ranges. Red lines shows the actual data ranges, blue lines show the configured ranges. 

![test](https://user-images.githubusercontent.com/5449929/200851259-7d7129d5-c135-424c-b6bb-a5f6fdda76a7.png)

