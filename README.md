# Monte Carlo Localization
**Members** : <a href="https://github.com/vegovs">Vegard Bergsvik Øvstegård</a>

**Supervisors** : <a href="https://www.mn.uio.no/ifi/personer/vit/jimtoer/">Jim Tørresen</a>

## Description

This repository contains a crude implementation the Monte Carlo Localization algorithm, a.k.a particle filter.
It is used mainly for simulation and testing.

![](data/Peek%202020-11-26%2000-05.gif)

As mentioned, this is a crude implementation. It does have a form of dynamic particle
initialization, but is in no way optimized. Red circle is the UAV/Robot, and black dots with
lines are particles. The particle dots increase in size and turn green when they are likely ti
bee in the same location as the UAV/Robot.

## Dependencies
* [Python](https://www.python.org/) (version 3.8)
* [Pip](https://virtualenv.pypa.io/en/latest/)
* [virtualenv](https://virtualenv.pypa.io/en/latest/) or:

## Installation

```console
git clone https://github.com/gil-uav/crude_mcl.git
```

#### virtualenv

```console
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage
```
python monte_carlo_localization.py
```

Press `r` to reset the simulation, and `space` to pause.

One can also tinker with the global parameters in the top of [monte_carlo_localization.py
](monte_carlo_localization.py)
