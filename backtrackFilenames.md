How to backtrack single cells:

- prepareData.py in cellProfilerAlgorithms reads
path_to_data = '/Volumes/MoritzBertholdHD/CellData/featureExtractionCellProfiler/extracted_data_ex1/'

- in extracted_data_ex1/data_ch1_no_zeros.csv the features are listed including the imagename with plate and cell id.
- dimensions of "names" and "data" are the same! Hopefully also ordered.
- the images can then be found in Experiments/Ex1/LabeledImages/full_ex1_no_zeros/++/--, depending on the class of the image.
