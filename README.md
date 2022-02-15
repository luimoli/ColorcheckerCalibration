# Calibration

## color checker
The color checker we use could be regarded as a 4 x 6 matrix, which should form an image of shape [4, 6, 3].

Assume Color Correction Matrix to be A, 
let P be a reference color checker matrix (24 x 3) and O be a color checker matrix to correct (24 x 3).  
we calculate a 3x3 matrix A which approximate the following equation.  
`P = [O 1] A`


## Data
We have to prepare color checker patch data as csv format.
There are example data in `data` directory.
- `data/measure.csv`
- `data/real.csv`  

## Prerequisites
- Python 3.8.11
- Pytorch 1.9
- opencv-python 4.5.3.56
- numpy 1.20.1

## Usage
