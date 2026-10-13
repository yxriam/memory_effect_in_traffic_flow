# Shanghai taxi GPS raw data (not distributed)

Place the 4 316 per-taxi trajectory files of the SUVnet 2007-02-20 collection in `Taxi_070220/`
(one file per taxi, columns: id, time, lon, lat, speed, heading, status).
The notebook reads them with `DATA_DIR = './data/Taxi_070220'` only when `RUN_COMPUTE = True`.

- Source: SUVnet project, Shanghai Jiao Tong University. Check the data provider's terms before redistributing.
- Size: ≈ 350 MB raw (≈ 43 MB zipped, ≈ 22 MB with xz/7z).
- Trajectory bounding box used for the global grid: lat 30.902–31.408, lon 121.1053–121.8031.
