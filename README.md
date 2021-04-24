# ARN sequential Pipeline

## Introduction


## Installation
Requirements:

- Python 3.6 or greater
- Numpy

## Usage
The script has two main options (MULTIPROCESSING and SINGLE PROCESSOR). The code for the multiprocessing option is commented in the script. To execute the script open a terminal (or command line depending on the OS) and write the following:

For Mac OS and Linux
```bash
python3 pipeline.py
```
For Windows
```bash
python pipeline.py
```

If you want to run the code with different dimensions or with more values change
```python
n = np.array([260,20]) # number of rows
m = np.array([40,20])  # number of columns 
rho1 = np.array([0.1]) # the value of rho for the first group (or control)
rho2 = np.array([0.2]) # the value of rho for the second group (or patients)
b_m = 20    	       # mean value
```

The script will print in the terminal the results of the misclassification mean error for all the combinations of the n and m values.