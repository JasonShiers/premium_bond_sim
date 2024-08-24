# NS&I Premium Bonds Monte Carlo simulator (Python)

## Description

Contains a script to perform a Monte Carlo simulation of Premium Bond draws. Using the winning odds and prize matrix published by NS&I from time to time, a set of 100k bonds is monitored over 12 (annual) draws for batches of 50k simulations.
A list of (bond number, prize) combinations is saved to a pkl file for analysis.

The analysis tool loads and combines pkl files, then transforms and analyses the data to generate median and quartile rates of return for various sized bond holdings. A plot of return vs holding size is generated.

## Prerequisites

- Python 3.12+
- Required Python packages
	- pandas
	- numpy
	- matplotlib
	- seaborn
