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
import pickle
import numpy as np

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

BONDS_VALUE = 100_000

SIM_START = 00_000
SIM_END = SIM_START + 50_000

RNG: np.random.Generator = np.random.default_rng()


def prize_draw(bonds_value: int) -> list[tuple[int, int]]:
    """ Simulate single prize draw  with bondsValue bonds
        returns list of winning (bond, prize) """
    # Check each bond for winning condition (value = 0)
    outcomes: np.ndarray = RNG.choice(WIN_CHOICES, size=bonds_value)
    winners: np.ndarray = np.where(outcomes == 0)[0]

    # randomly sample prizes (without replacement) each winner
    prizes: np.ndarray = RNG.choice(PRIZES, replace=False, size=len(winners))
    # return list of winning bonds and assigned prizes
    return list(zip(winners, prizes))


def annual_prizes(bonds_value: int) -> list[tuple[int, int]]:
    """ Simulate prizes over a year """
    winnings = []

    for _ in range(12):
        winnings.extend(prize_draw(bonds_value))
    return winnings


def set_data_matrix(sim: int) -> list[tuple[int, int]]:
    """ Wrapper function for parallel processing """
    return annual_prizes(BONDS_VALUE)


pool = Pool(processes=6)
startTime = perf_counter()
dataMatrix = pool.map(set_data_matrix, range(SIM_START, SIM_END))
endTime = perf_counter()
print(f"Completed in {endTime - startTime:0.4f} s")

with open('premiumBondDataMatrix'+str(SIM_START)+'.pkl', 'wb') as file:
    pickle.dump(dataMatrix, file)
