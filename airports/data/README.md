# BTS DB1B data (not distributed)

```
airports/data/
  original_data/db1b.public.YYYYMM.asc   # quarterly DB1B Origin & Destination Survey extracts (public, U.S. DOT)
  sequences_split/YYYYMM/YYYYMM_<n>.txt  # produced by pre_process/airports_sequences_split.py: paths with n airports
  predict1.txt, predict2.txt, …          # Markov rollouts written by markov_chain/AirportsMarkovChain.py
```

Download: https://www.transtats.bts.gov/Tables.asp?DB_ID=125 (Airline Origin and Destination Survey, DB1B).
The paper uses 2024 Q1 (511 407 valid paths, 436 airports, mean path length 3.39).
