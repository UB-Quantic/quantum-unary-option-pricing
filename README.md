# *[Quantum unary approach to option pricing](https://arxiv.org/abs/1912.01618)*

#### Sergi Ramos-Calderer, Adrián Pérez-Salinas, Diego García-Martín, Carlos Bravo-Prieto, Jorge Cortada, Jordi Planagumà, and José I. Latorre.


This is a repository for all code written for the article "*Quantum unary approach to option pricing*. 

It gives numerical simulations of the quantum classifier in [arXiv: 1912.01618](https://arxiv.org/abs/1912.01618).

All code is written Python. Libraries required:

  - matplotlib for plots
  - numpy, os, scipy
  - qiskit

##### Files included:
  - [aux_functions.py](https://github.com/UB-Quantic/quantum-finance/blob/master/aux_functions.py): File with some auxiliary functions that simplify the code
  - [binary.py](https://github.com/UB-Quantic/quantum-finance/blob/master/binary.py): File with all pieces required for performing calculations in the binary representation
  - [errors.py](https://github.com/UB-Quantic/quantum-finance/blob/master/errors.py): Class with all different calculations implemented
  - [unary.py](https://github.com/UB-Quantic/quantum-finance/blob/master/unary.py): File with all pieces required for performing calculations in the binary representation
  - [main.py](https://github.com/UB-Quantic/quantum-finance/blob/master/main.py): This is the only file one needs to change. Everything can be set up there: number of bins, binary and unary, parameters defining the model, what functions are to be run... The only thing one has to do is to run this file. Comments on what the lines do are written in the source.
  - [noise_mapping.py](https://github.com/UB-Quantic/quantum-finance/blob/master/noise_mapping.py): Generates noise maps for `qiskit`
  - [results.tar.gz.py](https://github.com/UB-Quantic/quantum-finance/blob/master/results.tar.gz): Tar file with results as computed in the paper. Not needed, but may save some time

##### How to cite

If you use this code in your research, please cite it as follows:

Ramos-Calderer, S., Pérez-Salinas, A., García-Martín, D., Bravo-Prieto, C., Cortada, J., Planagumà, J., & Latorre, J. I. (2019). Quantum unary approach to option pricing. arXiv preprint arXiv:1912.01618.

BibTeX:

`@misc{ramoscalderer2019quantum,\
    title={Quantum unary approach to option pricing},\n
    author={Sergi Ramos-Calderer and Adrián Pérez-Salinas and Diego García-Martín and Carlos Bravo-Prieto and Jorge Cortada and Jordi Planagumà and José I. Latorre},\n
    year={2019},\n
    eprint={1912.01618},\n
    archivePrefix={arXiv},\n
    primaryClass={quant-ph}\n
}
`



