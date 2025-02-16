"""Premium Bond Monte Carlo simulator.

@author: Jason Shiers

Generates files for a Monte Carlo simulation of Premium Bond draws
Simulates annual draws for chunks of 50,000 simulations per data file
Uses multiprocessing and numpy random functions for optimal efficiency
"""

from multiprocessing import Pool
from time import perf_counter

import numpy as np
import polars as pl
from numpy.typing import NDArray

PRIZE_MATRIX = {
    25: 1_475_218,
    50: 2_190_094,
    100: 2_190_094,
    500: 54_807,
    1_000: 18_269,
    5_000: 1_747,
    10_000: 874,
    25_000: 350,
    50_000: 175,
    100_000: 87,
    1_000_000: 2,
}

PRIZES: NDArray[np.int32] = np.repeat(
    tuple(PRIZE_MATRIX.keys()), tuple(PRIZE_MATRIX.values()),
).astype(np.int32)

WINNING_ODDS = 21_000
WIN_CHOICES: NDArray[np.int16] = np.arange(WINNING_ODDS).astype(np.int16)

# Set up a random generator for prize_draw function
RNG: np.random.Generator = np.random.default_rng()

BONDS_VALUE = 100_000  # Maximum holding size of interest


def prize_draw(bonds_value: int) -> tuple[NDArray[np.int16], NDArray[np.int32]]:
    """Simulate single prize draw  with bondsValue bonds.

    Returns:
        list of winning (bond, prize)

    """
    # Check each bond for winning condition (value = 0)
    outcomes: NDArray[np.int16] = RNG.choice(WIN_CHOICES, size=bonds_value)
    winners: NDArray[np.int16] = np.where(outcomes == 0)[0]

    # randomly sample prizes (without replacement) each winner
    prizes: NDArray[np.int32] = RNG.choice(
        PRIZES, replace=False, size=len(winners),
    )
    # return list of winning bonds and assigned prizes
    return winners, prizes


def monte_carlo_sim(sim: int) -> pl.DataFrame:
    """Perform one simulation of a monte carlo experiment.

    Returns:
        DataFrame of winning bond and prize

    """
    winners, prizes = prize_draw(BONDS_VALUE)
    schema = {"bond": pl.Int32, "prize": pl.Int32}
    return pl.DataFrame(zip(winners, prizes), schema=schema,
    ).with_columns(pl.lit(sim).alias("sim").cast(pl.Int32))


pool = Pool(processes=6)
start_time = perf_counter()
monte_carlo_results = pool.map(monte_carlo_sim, range(6_000_000))
end_time = perf_counter()
print(f"Completed in {end_time - start_time:0.2f} s")  # noqa: T201
start_time = perf_counter()
experiment = pl.concat(monte_carlo_results)
end_time = perf_counter()
print(f"Aggregated results in {end_time - start_time:0.2f} s")  # noqa: T201
start_time = perf_counter()
experiment.write_parquet(
    "premium_bond_6M_sim_202408.parquet", compression_level=22,
)
end_time = perf_counter()
print(f"Wrote output file in {end_time - start_time:0.2f} s")  # noqa: T201
