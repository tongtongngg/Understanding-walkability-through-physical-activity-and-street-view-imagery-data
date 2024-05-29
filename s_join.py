from Streetedge_load import *
import pandas as pd

# create df_lines and df_points GeoDataFrames
df_lines = data 

image_location = pd.read_csv('image_locations.csv')
print(image_location)

df_points = gpd.GeoDataFrame(image_location, geometry=gpd.points_from_xy(image_location.lat, image_location.lng))
df_points = df_points.drop(columns=['lat', 'lng'])
print(df_points)
# copy df_lines to create df_polygons
df_polygons = df_lines.copy()

# set the crs of df_polygons and df_points to match df_lines
df_polygons.crs = df_lines.crs
df_points.crs = df_lines.crs

# buffer the geometry of df_polygons
df_polygons['geometry'] = df_polygons['geometry'].buffer(0.0001)


# perform the spatial join
pointInPoly = gpd.sjoin(df_points, df_polygons,how='left',predicate='within')

#save the dataframe
pointInPoly.to_csv('sjoin_data.csv')

# print(pointInPoly)  
# pointInPoly.plot()
# plt.show()

