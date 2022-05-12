# ColorChecker Calibration

## ColorChecker
The colorchecker we use contains 4 x 6 special standard color patches and could be regarded as a 4 x 6 matrix, which form an image of shape [4, 6, 3].

## CCM (Color Correction Matrix)
Designed to use after *AWB* and before *nonlinear-tranform (gamma)*.

CCM is calculated based on raw-sensor-rgb of the shooted colorchecker and corresbonding ground-truth-rgb. 

Assume Color Correction Matrix to be A, 
let P be a reference colorchecker matrix (24 x 3) and O be a colorchecker matrix to correct (24 x 3).  
we calculate a 3x3 matrix A which approximate the following equation until the best A founded.  
`P = [O 1] A`

CCM varies when illuminants and nonlinear-function changes.

![image.png](https://raw.githubusercontent.com/luimoli/mmgraph/main/pic/image2022-2-21_16-12-32.png)


## Usage
### Data
Set color checker patch data as csv format.
There are example data in `data` directory.
- `data/measure_xx.csv`
- `data/real_xx.csv`  

### Prerequisites
- Python 3.8.11
- opencv-python 4.5.3.56
- numpy 1.20.1

### Usecase
```

```

