# %%
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt



# %%
# Load the CSV file into a DataFrame
df1 = pd.read_csv('sjoin_data_kbh+amager.csv')
df2 = pd.read_csv('sjoin_data_glostrup.csv')
df = pd.concat([df1, df2])
print(df)

# Get a list of the image IDs from the image folder
image_folder = '/home/s214626/billeder'  
image_ids = [filename.split('.')[0] for filename in os.listdir(image_folder) if filename.endswith('.jpg')]

# Filter the DataFrame to keep only the rows with matching image IDs
df = df[df['image_id'].astype(str).isin(image_ids)]

# Reset the index of the DataFrame
df = df.reset_index(drop=True)

# Filter the DataFrame to keep only the rows with non-null edgeUID values
df = df.dropna(subset=['edgeUID'])

# Reset the index of the DataFrame
df = df.reset_index(drop=True)

df = df.drop('Unnamed: 0', axis=1)

# add image path
file_extension = '.jpg'  # Replace with the actual file extension of your images

def get_image_path(image_id):
    return os.path.join(image_folder, str(image_id) + file_extension)

df['image_path'] = df['image_id'].map(get_image_path)

# Save the modified DataFrame back to a CSV file
df.to_csv('modified_file.csv', index=False)

# %%
df2 = pd.read_csv('7f42d69dd7bd00b04081b9cbd0d05ab4e5f87353dcd7364c34b8e7c74d42109d-1676891223740.csv')

sum_df = df2.groupby('edge_uid')['total_trip_count'].sum().reset_index()

print(sum_df)

# %%
sum_df.to_csv('total_trip_counts.csv', index=False)

# %%
merged_df = pd.merge(df, sum_df, left_on='edgeUID', right_on='edge_uid', how='left')
merged_df = merged_df.drop('edge_uid', axis=1)
merged_df.dropna(subset=['total_trip_count'], inplace=True)
merged_df.to_csv('final_data.csv', index=False)
print(merged_df)

# %% [markdown]
# ## 


