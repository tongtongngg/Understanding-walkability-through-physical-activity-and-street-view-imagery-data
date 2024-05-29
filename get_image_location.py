import requests
import time
import os
import pandas as pd
access_token = "MLY|5652096898253217|954722f6082bab20a3dfe6df116d6d44"

def get_image_location(image_key):
    endpoint = f"https://graph.mapillary.com/{image_key}"
    params = {"fields": "computed_geometry", "access_token": access_token}
    response = requests.get(endpoint, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if "computed_geometry" in data:
            lat, lng = data["computed_geometry"]["coordinates"]
            return lat, lng
        else:
            print(f"No computed geometry found for image {image_key}")

    else:
        print(f"Request failed: {response.text}")
    
    return None


# read image ids from file
with open("image_ids.txt") as f:
    image_keys = [line.strip() for line in f]

#get image location
latitude = []
longitude = []
for image_key in image_keys:
    location = get_image_location(image_key)
    if location is not None:
        if location[0] is not None and location[1] is not None:
            latitude.append(location[0])
            longitude.append(location[1])
    else:
        #print(f"Location not found for image {image_key}")
        #delete image id from list
        image_keys.remove(image_key)
    # if location:
    #     print(f"Image {image_key} is located at ({location[0]}, {location[1]})")
    time.sleep(0.1)
            
# Save it in a dataframe      
image_location = pd.DataFrame(
    {'imagekey': image_keys,
     'lat': latitude,
    'lng': longitude
    })
print(image_location)