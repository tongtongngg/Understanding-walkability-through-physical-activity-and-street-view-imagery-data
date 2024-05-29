import mercantile
import mapbox_vector_tile
import requests
import json
import os
from vt2geojson.tools import vt_bytes_to_geojson
from PIL import Image
from io import BytesIO
from IPython.display import HTML, display
import folium

# gade: 12.566042,55.652943,12.696419,55.687891
# USA:  -80.13423442840576,25.77376933762778,-80.1064238357544,25.888608487732198

# Lille område: 12.618476,55.658367,12.620955,55.658561

# Ved nørreport. 12.560036, 55.682185,  12.573124, 55.685577

# id:  5652096898253217

# AMAGER OG KBH: 12.436781, 55.545432, 12.696966, 55.731815

tile_coverage = 'mly1_public'
tile_layer = "image"

# Mapillary access token -- user should provide their own
access_token = 'MLY|5652096898253217|954722f6082bab20a3dfe6df116d6d44'

# A few coordinates
#55.694833, 12.566544, 55.683319, 12.588378
#55.697357, 12.503808 , 55.657248, 12.602819
# a bounding box in [east_lng,_south_lat,west_lng,north_lat] format

west, south, east, north = [12.530883, 55.740750,  12.559771, 55.755021]
# dragør : 55.596243, 12.647799, 55.581821, 12.682225
# get the list of tiles with x and y coordinates which intersect our bounding box
# MUST be at zoom level 14 where the data is available, other zooms currently not supported
tiles = list(mercantile.tiles(west, south, east, north, 14))

# make a map called pictures
#os.makedirs('billeder')

# loop through list of tiles to get tile z/x/y to plug in to Mapillary endpoints and make request
for tile in tiles:
    tile_url = 'https://tiles.mapillary.com/maps/vtp/{}/2/{}/{}/{}?access_token={}'.format(tile_coverage, tile.z, tile.x, tile.y, access_token)
    response = requests.get(tile_url)
    data = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z, layer=tile_layer)

    # loop through the features in the data to extract the image URLs and save the images locally
    for feature in data['features']:
        # get lng,lat of each feature
        lng = feature['geometry']['coordinates'][0]
        lat = feature['geometry']['coordinates'][1]

        # ensure feature falls inside bounding box since tiles can extend beyond
        if lng > west and lng < east and lat > south and lat < north:
            # create a folder for each unique sequence ID to group images by sequence
            sequence_id = feature['properties']['sequence_id']
            

            # request the URL of each image
            image_id = feature['properties']['id']
            header = {'Authorization': 'OAuth {}'.format(access_token)}
            url = 'https://graph.mapillary.com/{}?fields=thumb_2048_url'.format(image_id)
            r = requests.get(url, headers=header)
            data = r.json()
            image_url = data['thumb_2048_url']

            # save each image with ID as filename to directory by sequence ID
            with open('{}/{}.jpg'.format("billeder", image_id), 'wb') as handler:
                image_data = requests.get(image_url, stream=True).content
                handler.write(image_data)
                
