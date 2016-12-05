# Data Preprocessing and Preparation for Pipeline

## 1st Step:
- use Matlab script convert.m to convert all .DIBs in Folder to .TIFs

## 2nd Step:
- use cellSegmentor.py to load .TIFs for each image and segment the cells.

Ideas for segmentation:
- use zero padding
- use round label kernel
- preprocess the mask using morphological opperations
- segment including also image information

Open points for discussion:
- are labels somewhere hidden in the data?
- how to resize the cells to uniform size?
- multiply masks on all channels or just cut rectangles?
- data format lmdb for caffe?

Erosion:
- does not work well, leaves tiny dots, too brutal

Opening:
- use large kernel (10x10), single iteration
- seperates blobs well!
- zero padding needed to get full size