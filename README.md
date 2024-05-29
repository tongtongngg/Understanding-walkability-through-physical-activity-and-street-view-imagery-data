# KID-projekt
"""This study aims to shed light on urban planning issues and identify which features of a street are
likely to attract more pedestrian activity. To obtain the connection between features and the number of
walkers, data from Mapillary was downloaded, and using spatial joining methods it was merged with
Metro Strava data, to create a thorough dataset for our research.

Three different areas; Nørreport, Gentofte, and Dragør were chosen to be representative of a wider
range of residential areas. Our transfer-learned CNN model was used in these areas to produce pre-
dictions on the number of walkers each image would have. From the image representation vectors, we
could run PCA for dimensionality reduction and perform K-means clustering, to see if it was possible
to visually notice any similarities within each cluster.

We found that the model performed the best with the images from Dragør, with a test error of
around 0.711. After our models’ predictions, we noticed some trends in the images that had the high-
est prediction. After getting results from the model, for each area of Copenhagen, we performed PCA
and k-means clustering. From this, we found features that appeared in different clusters from different
areas. We noticed that clusters including good road and transport infrastructure together with green-
ery and building generally resulted in more pedestrian and human activity, while clusters containing
broader roads such as highways resulted in limited activity.

Doing the project we found that the data we received from Strava Metro, was likely not represen-
tative of the entirety of Copenhagen, and instead had a bias toward runners. We found images from
highly active sites and saw values that did not match reality. This meant that our model was not the
best at predicting amount of walkers, and did not consider many aspects such as metro stations that
could make predictions higher, and instead focused on where it was optimal to run.

With continued refinement and improvements, our model has the potential to serve as a valuable
tool for predicting pedestrian activity. The insights gained from these predictions can inform decisions
related to urban planning and more. By understanding the connection between human mobility and
the environment, we can work towards creating more pedestrian-friendly and efficient surroundings in
areas with lower pedestrian activity. This can contribute to the development of walkable communities
and improve the overall quality of urban life.
"""

