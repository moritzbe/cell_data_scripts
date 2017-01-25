# Data Preprocessing and Preparation for Pipeline

## General Overview: 1st Step:
- use Matlab script convert.m to convert all .DIBs in Folder to .TIFs

## 2nd Step:
- use cellSegmentor.py to load .TIFs for each image and segment the cells.
- adjust parameters in settings
- all operations are performed in "#Iterations"
- single cells are stored in folders ++/+-/-+/--

## 3rd step: Run Matlab Script: 
- hucho_make_montages.m, it prepares the data for cellprofiler
- stores data in montages file in new folders (+-/--/-+/++)

## 4th step: CellProfiler:
- open cellProfiler 3 (currently beta, but opens .cpprj files) and open the Pipeline
- drag and drop all folders (+-/--/-+/++) into the cellProfiler panel
- select filter ch1
- determine output folder
- run it and repeat for ch2, ch3, ch4 after step5 was completed

## 5th step: Run Matlab Script
- run matlab script hucho_prepare_data_with_filenames.m
- change number of montages in each class
- change input/output folders
- delete old data to pursue with ch2, ch3, ch4
- the output should be 4 .csv files (4 channels)

## 6th step: Python script to prepare .npy data
- change folder to cellProfilerAlgorithms/prepareData.py
- specify inputs and run
- output are data.npy and labels.npy for the corresponding experiment


Ideas for segmentation:
- use zero padding
- use round label kernel
- preprocess the mask using morphological opperations
- segment including also image information

Open points for discussion:
- how to resize the cells to uniform size?
- multiply masks on all channels or just cut rectangles?
- data format lmdb for caffe?

Erosion:
- does not work well, leaves tiny dots, too brutal

Opening:
- use large kernel (10x10), single iteration
- seperates blobs well!
- zero padding needed to get full size

## ToDos:
- cut morph. operations
- check if there are zeros in cells

Distribution of Labels gives: 
- Label Frequencies are: [ 3376  1433  8270 11112  1312]