# Modified Bases Time Series Analysis
Official implementation of our paper "Expanding the Molecular Alphabet of DNA-Based Data Storage Systems with Nanopore Readouts".

Programming language: Python 3.7. Tested on operating systems: Windows 10, CentOS 7.7.1908.

Our method can be divided into two components: extracting signals of interest from raw fast5 files, and perform neural network classification on time series.

# Signal Extraction

The raw fast5 files for all 77 tetramers are stored in [link]. We also store the final results after signal extraction step as `npy` files in [link].

The overall pipeline of this component is shown in the following figure. Please refer to **Figure S6** for more details.

![image info](./signal_extraction_pipeline.png)

