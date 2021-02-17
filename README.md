# DynPageRank

The scripts in this repository can be used to compute the TempoRank centrality
measure for temporal networks. There are two scripts, corresponding to two 
different computation methods:

* `temporank_deterministic.py` computes the TempoRank vectors exactly using
matrix algebra. 
* `temporank_stochastic.py` computes an approximation to the TempoRank vectors
using a stochastic approximation by temporal random walks.

Both scripts takes as input a *movement file*, formatted as a `.csv` file, in
which each directed time-stamped edge is represented by a row:

```
date1,origin1,destination1
date2,origin2,destination2
date3,origin3,destination3
...
```

## Deterministic algorithm

Using this script requires the 
[Petsc4py](https://www.mcs.anl.gov/petsc/petsc4py-current/docs/) and 
[Slepc4py](https://pypi.org/project/slepc4py/) libraries. The basic
syntax
for computing TempoRank using the deterministic method is as
follows:

```
python temporank_deterministic.py input_mvtfile.csv --version output_file.csv
```

Optional parameters for `temporank_deterministic.py` are:

* `--laziness` (default q=0) sets the per-timestep probability for a random
walker to remain in its current node.
* `--teleport` (default d=0.01) sets the per-timestep probability for a random
walker to move to a uniform node in the network, conditionally on it not being
lazy.
* `--verbose` (default False) selects whether to display information about the
network and the current number of iterations.

## Stochastic algorithm

The basic syntax for computing TempoRank using temporal random walks is as 
follows:

```
python temporank_stochastic.py input_mvtfile.csv 1000000 output_file.csv
```

The first parameter is a movement file, formatted as described above. The second
parameter `num_iter` corresponds to the number of steps taken by the
random walkers before stopping, and the third to the output file, which will be
formatted as a `.csv` file:

```
node1,TR1,q,d,n
node2,TR2,q,d,n
...
```

Optional parameters for `temporank_stochastic.py` are:

* `--laziness` (default q=0) sets the per-timestep probability for a random
walker to remain in its current node.
* `--teleport` (default d=0.01) sets the per-timestep probability for a random
walker to move to a uniform node in the network, conditionally on it not being
lazy.
* `--convergence` (default False) logs the TempoRank vector at increments of
`num_iter`/100 steps in order to verify its convergence to the deterministic
limit.
* `--version` (default 2) selects whether to compute the TempoRank as described
in (Rocha, Masuda 2014) (`--version 1`) or the out-TempoRank of (Hoscheit et al.
2021).
* `--verbose` (default False) selects whether to display information about the
network and the current number of iterations.
* `--date` (default False) is to be used if the timestamps (the first column of
the input movement file) are in date format, using the ISO 8601 standard 
(YYYY-MM-DD).