import pylab as plt
import tqdm as tqdm
import pandas as pd
from herbie import FastHerbie
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings('ignore')
import zarr
from IPython.utils import io

DATES = pd.date_range('2021-03-09 00:00', '2021-03-10 00:00', freq='1H')

# Setup Herbie class for surface fields
# The save_dir optional argument allows you to specify where herbie archives the data
H = FastHerbie(DATES, model="hrrr", product="sfc") #save_dir='/Users/pres026/Research/Datasets/hrrr_data/hrrr')

#Setup Herbie class for native level fields 
Hnat = FastHerbie(DATES, model="hrrr", product="nat")# save_dir='/Users/pres026/Research/Datasets/hrrr_data/hrrr')

surface_name = {} 
surface_name['VIL'] = "VIL"
surface_name['SST'] = ":TMP:surface"  
surface_name['T2m'] ="TMP:2 m"
surface_name['QV2m'] ="SPFH:2 m"
surface_name['U80'] = "UGRD:80 m"
surface_name['V80'] = "VGRD:80 m"
surface_name['U10'] = "UGRD:10 m"
surface_name['V10'] = "VGRD:10 m"
surface_name['PSFC'] = "PRES:surface"
surface_name['LHTFL'] = "LHTFL:surface"
surface_name['SHTFL'] = "SHTFL:surface"
surface_name['DSWRF'] = "DSWRF:surface"
surface_name['LCDC'] = "LCDC"
surface_name['WIND'] = "WIND"
surface_name['HPBL'] = "HPBL"
surface_name['LAND'] = "LAND"
surface_name['DLWRF'] = "DLWRF:surface"

for name, regex in tqdm.tqdm(surface_name.items(), desc="Downloading Surface Data"):
    H.download(regex)
    
level_name = {} 
level_name["U"] = ":(?:U)GRD:[0-9]+ hybrid"
level_name["V"] = ":(?:V)GRD:[0-9]+ hybrid"
level_name["P"] = ":(?:PRES):[0-9]+ hybrid"
level_name["T"] = ":(?:TMP):[0-9]+ hybrid"
level_name["QV"] = ":(?:SPFH):[0-9]+ hybrid"
level_name["QC"] = ":(?:CLMR):[0-9]+ hybrid"
level_name["QI"] = ":(?:CIMIXR):[0-9]+ hybrid"
level_name["Z"] =":(?:HGT:)[0-9]+ hybrid"
print(" ")
for name, regex in tqdm.tqdm(level_name.items(), desc="Downloading Native Grid Data"):
    Hnat.download(regex)
    
