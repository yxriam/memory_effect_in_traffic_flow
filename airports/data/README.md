# BTS DB1B data (not distributed)

```
airports/data/
  original_data/db1b.public.YYYYMM.asc   # quarterly DB1B Origin & Destination Survey extracts (public, U.S. DOT)
  sequences_split/YYYYMM/YYYYMM_<n>.txt  # written by airports_sequences_split.py: itineraries with n airports
  sankey/city_sankey_order{1,2}.{json,html}   # written by city_markov_sankey.py
```

Download: https://www.transtats.bts.gov/Tables.asp?DB_ID=125 (Airline Origin and Destination Survey, DB1B).
The analysis uses 2024 Q3 and Q4 (`db1b.public.202409.asc`, `db1b.public.202412.asc`): 5 114 707 and 5 186 452 itineraries, 452 and 450 airports, mean length 3.39 and 3.37 airports.
Model weights and rollouts of `lstm_rnn_predict.ipynb` go to `airports/output/` (git-ignored).
