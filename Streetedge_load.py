
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

show_bounding_box = True

# Get the absolute path of the current directory
dir_path = os.getcwd()

# Search for the shapefile in the current directory and its subdirectories
for root, dirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith(".shp"):
            shp_path = os.path.join(root, file)
            break

# Load the shapefile using GeoPandas
data = gpd.read_file(shp_path)

if show_bounding_box:
    # 12.618476, 55.658367, 12.620955, 55.668561

    # Define the bounding box coordinates
    bbox = box(12.662613, 55.583994,  12.672050, 55.594517)
    bbox = gpd.GeoSeries(bbox)

    # Overlay the bounding box on the map
    ax = data.plot(figsize=(10, 10), alpha=0.5)
    bbox.plot(ax=ax, color='red', linewidth=2)
    plt.show()

else:
    #plot Strava data without bounding box
    data.plot()
    plt.show()
print("done")
