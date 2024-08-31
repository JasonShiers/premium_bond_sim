#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:57:15 2022

@author: Jason Shiers

Premium Bond Monte Carlo simulator
---------------------------------
Generates files for a Monte Carlo simulation of Premium Bond draws
Simulates annual draws for chunks of 50,000 simulations per data file
Uses multiprocessing and numpy random functions for optimal efficiency
"""

from multiprocessing import Pool
from time import perf_counter
import numpy as np
import polars as pl

PRIZE_MATRIX = {
    25: 1_475_218, 50: 2_190_094, 100: 2_190_094, 500: 54_807,
    1_000: 18_269, 5_000: 1_747, 10_000: 874, 25_000: 350,
    50_000: 175, 100_000: 87, 1_000_000: 2
    }

PRIZES: np.ndarray = np.repeat(
    tuple(PRIZE_MATRIX.keys()), tuple(PRIZE_MATRIX.values())
    ).astype(np.int32)

WINNING_ODDS = 21_000
WIN_CHOICES: np.ndarray = np.arange(WINNING_ODDS).astype(np.int16)

# Set up a random generator for prize_draw function
RNG: np.random.Generator = np.random.default_rng()

BONDS_VALUE = 100_000  # Maximum holding size of interest


def prize_draw(bonds_value: int) -> tuple[np.ndarray, np.ndarray]:
    """ Simulate single prize draw  with bondsValue bonds
        returns list of winning (bond, prize) """
    # Check each bond for winning condition (value = 0)
    outcomes: np.ndarray = RNG.choice(WIN_CHOICES, size=bonds_value)
    winners: np.ndarray = np.where(outcomes == 0)[0]

    # randomly sample prizes (without replacement) each winner
    prizes: np.ndarray = RNG.choice(PRIZES, replace=False, size=len(winners))
    # return list of winning bonds and assigned prizes
    return winners, prizes


def monte_carlo_sim(sim: int) -> pl.DataFrame:
    """ Perform one simulation of a monte carlo experiment """
    winners, prizes = prize_draw(BONDS_VALUE)
    schema = ({'bond': pl.Int32, 'prize': pl.Int32})
    df = pl.DataFrame(zip(winners, prizes), schema=schema)
    df = df.with_columns(pl.lit(sim).alias('sim').cast(pl.Int32))
    return df


pool = Pool(processes=6)
startTime = perf_counter()
monte_carlo_results = pool.map(monte_carlo_sim, range(6_000_000))
endTime = perf_counter()
print(f"Completed in {endTime - startTime:0.2f} s")
startTime = perf_counter()
df = pl.concat(monte_carlo_results)
endTime = perf_counter()
print(f"Aggregated results in {endTime - startTime:0.2f} s")
startTime = perf_counter()
df.write_parquet('premium_bond_6M_sim_202408.parquet', compression_level=22)
endTime = perf_counter()
print(f"Wrote output file in {endTime - startTime:0.2f} s")
