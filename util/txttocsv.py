import csv
import os
import pandas as pd

read_file = pd.read_csv (r'txt/Xeon_paddr_bin')
read_file.to_csv (r'txt/Xeon_paddr_bin.csv', index=None)

read_file = pd.read_csv (r'txt/Xeon_slice')
read_file.to_csv (r'txt/Xeon_slice.csv', index=None)

# Type shell "paste -d ',' txt/paddr_bin.csv txt/slice.csv > txt/data.csv"