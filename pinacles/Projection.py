import numpy as np
import cartopy.crs as ccrs
import pyproj

class LambertConformal:
    def __init__(self, earth_radius, lat_1, lat_2, lat_0, lon_0, proj_lon0):

        self.lat_0 = lat_0
        self.lon_0 = lon_0

        self.globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6370000, semiminor_axis=6370000)
        self.lcc = ccrs.LambertConformal(central_longitude=proj_lon0, central_latitude=lat_0,
                            standard_parallels=(lat_1, lat_2), globe=self.globe)

        self.c = ccrs.PlateCarree()

        # Transform from lat-lon to LCC (eastings and westing)
        self.transformer = pyproj.Transformer.from_crs(self.c, self.lcc)

        # Inverse transform from LCC (easting and westings) to lat lon
        self.transformer_i = pyproj.Transformer.from_crs(self.lcc, self.c)
        
        
        #Compute esting and westings for domain center
        self.center_e, self.center_n = self.transformer.transform(self.lon_0, self.lat_0)
        
        
        return



    def compute_xy(self, lon, lat):

        #Compute esting and westings for domain center
        e, n = self.transformer.transform(lon, lat)

    
        return e, n


    def compute_latlon(self, x, y):

        x += self.center_e
        y += self.center_n

        #Transform eastings and westings back to lat and lon
        lon, lat = self.transformer_i.transform(x, y)
        


        return lon, lat

