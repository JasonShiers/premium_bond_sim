# NS&I Premium Bonds Monte Carlo simulator (Python)

## Description

Contains a script to perform a Monte Carlo simulation of Premium Bond draws. Using the winning odds and prize matrix published by NS&I from time to time, the draw results for a set of 100k bonds is simulated over 6 million draws. A Polars DataFrame of (sim, bond, prize) combinations is saved to a parquet file for analysis.

The analysis tool loads the parquet file, then transforms and analyses the data to generate median and quartile rates of return for various sized bond holdings. A plot of return vs holding size is generated.

## Prerequisites

- Python 3.12+
- Required Python packages
	- polars
	- numpy
	- matplotlib
	- seaborn
