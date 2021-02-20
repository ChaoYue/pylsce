try: 
    import numpy as np
    print "numpy version: ",np.__version__
except ImportError:
    print "numpy not installed"

try:
    import matplotlib as mat
    print "matplotlib version: ",mat.__version__
except ImportError:
    print "matplotlib not installed"

try:
    import mpl_toolkits.basemap as bmp
    print "basemap version: ",bmp.__version__
except ImportError:
    print "basemap not installed"

try:
    import pandas as pa
    print "pandas version: ",pa.__version__
except ImportError:
    print "pandas not installed"

try:
    import netCDF4 as nc
    print "netCDF4 version: ",nc.__version__
except ImportError:
    print "netCDF4 not installed"

try:
    import scipy
    print "scipy version: ",scipy.__version__
except ImportError:
    print "scipy not installed"

try:
    import gdal
    print "gdal version: ",gdal.VersionInfo()
except ImportError:
    print "gdal not installed"

try:
    import statsmodels.api as sm
    print "statsmodels version: ",sm.__version__
except ImportError:
    print "statsmodels not installed"
