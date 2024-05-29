# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, Flatten
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm
import pandas as pd

y_train = np.load('/home/s214626/y_train2.npy')
y_test = np.load('/home/s214626/y_test2.npy')


train_image_features = np.load('/home/s214626/train_features2.npy')
test_image_features = np.load('/home/s214626/test_features2.npy')

trip = pd.read_csv('total_trip_counts.csv')
mean_value = trip['total_trip_count'].mean()
print(mean_value)

# %%
# Reshape the input features
train_image_features_2d = train_image_features.reshape(train_image_features.shape[0], -1)
test_image_features_2d = test_image_features.reshape(test_image_features.shape[0], -1)

# Build the model
input_shape = (train_image_features_2d.shape[1],)
image_input = Input(shape=input_shape, name='image_input')
hidden_layer = Dense(64, activation='relu')(image_input)
output = Dense(1, activation='linear')(hidden_layer)
model = Model(inputs=image_input, outputs=output)

# Define custom MAPE loss function
def mape_loss(y_true, y_pred):
    diff = tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), 1e-8, np.inf))
    return tf.reduce_mean(diff)

# Compile the model with MAPE loss
model.compile(loss=mape_loss, optimizer='adam')

# Define callbacks to calculate test loss after each epoch
class TestLossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        test_loss = self.model.evaluate(test_image_features_2d, y_test, verbose=0)
        #print(f'Test Loss after epoch {epoch + 1}: {test_loss:.4f}')

# Train the model
history = model.fit(train_image_features_2d, y_train, epochs=10, batch_size=32, verbose=1,
                    validation_data=(test_image_features_2d, y_test), callbacks=[TestLossCallback()])

# Evaluate the model using MAPE
model_predictions = model.predict(test_image_features_2d)


model_loss = mean_absolute_percentage_error(y_test, model_predictions)

# Print the final test loss
print("Final Test Loss:", model_loss)



# %%
import pandas as pd
# Convert the predictions to a dataframe
predictions_df = pd.DataFrame(model_predictions, columns=['predictions'])

# Save the predictions to a CSV file
predictions_df.to_csv('predictions2.csv', index=False)

# %%
# Plot the loss of the model
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], label='Training Loss')
plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('MAPE')
plt.title('Training and Test Loss: EfficientNetB0 Model')
plt.legend()
plt.show()


# %%
# Find the highest and lowest total trip count values
max_trip_count = y_train.max()
min_trip_count = y_train.min()

print("Highest Total Trip Count:", max_trip_count)
print("Lowest Total Trip Count:", min_trip_count)

# %%
max_trip_count = model_predictions.max()
print(max_trip_count )

# %%
import matplotlib.pyplot as plt

# Plot histogram
plt.hist(y_train/mean_value, bins=100)  # Adjust the number of bins as needed
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of y_train')
plt.show()


# %%
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.wkt import loads
from sklearn.model_selection import train_test_split

# Load the CSV data
data = pd.read_csv('final_data.csv')
# Preprocess the data

# Filter out rows with missing image paths
data = data.dropna(subset=['image_path'])
print(data.shape)
# Create a bounding box
bbox = box(12.530883, 55.740750,  12.549771, 55.755021)

# Convert the geometry strings to Point objects
data['geometry'] = data['geometry'].apply(lambda x: loads(x))

# Convert the data to a GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry='geometry')

# Filter out data outside the bounding box
data = gdf[gdf.geometry.within(bbox)]
print(data.shape)

# Split the dataset into training and testing sets
X = data['image_path']
y = data['total_trip_count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define file paths for saving the features

# %%
print(data)

# %%
import numpy as np
import matplotlib.pyplot as plt

# Get the indices of the top 50 highest predictions
top_indices = np.argsort(model_predictions.flatten())[::-1][:100]

# Create a set to store unique combinations of edgeUID and image_id
unique_combinations = set()
unique_image = set()
unique_edge = set()

# Plot the top 9 images with highest predicted values from different edgeUID and image_id
plt.figure(figsize=(12, 8))
subplot_index = 1
for index in top_indices:
    i= data[(data['image_path'] == X_test.iloc[index]) & (data['total_trip_count'] == y_test.iloc[index])].index[0]
    image_path = X_test.iloc[index]
    if image_path not in unique_image:
        unique_image.add(image_path)
        edge_id= data.loc[i, 'edgeUID']
        if edge_id not in unique_edge:
            unique_edge.add(edge_id)
            combination = (edge_id, image_path)
            unique_combinations.add(combination)

            predicted_value = model_predictions[index]
            actual_value = y_test.iloc[index]
            image = plt.imread(image_path)

            plt.subplot(3, 3, subplot_index)
            plt.imshow(image)
            np.set_printoptions(precision=5, suppress=True)
            plt.title(f"Predicted: {predicted_value/mean_value}\nActual: {'{:.5g}'.format(actual_value/mean_value)}")
            plt.axis('off')

            subplot_index += 1

    if len(unique_combinations) == 9:
        break
plt.suptitle("Top 9 Images with Highest Predicted Values (Unique edgeUID and image_id)", fontsize=16)
plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# Get the indices of the bottom 50 predictions
bottom_indices = np.argsort(model_predictions.flatten())[:50]

# Create a set to store unique combinations of edgeUID and image_id
unique_combinations = set()
unique_image = set()
unique_edge = set()

# Plot the bottom 9 images with highest predicted values from different edgeUID and image_id
plt.figure(figsize=(12, 8))
subplot_index = 1
for index in bottom_indices:
    i= data[(data['image_path'] == X_test.iloc[index]) & (data['total_trip_count'] == y_test.iloc[index])].index[0]
    image_path = X_test.iloc[index]
    if image_path not in unique_image:
        unique_image.add(image_path)
        edge_id= data.loc[i, 'edgeUID']
        if edge_id not in unique_edge:
            unique_edge.add(edge_id)
            combination = (edge_id, image_path)
            unique_combinations.add(combination)

            predicted_value = model_predictions[index]
            actual_value = y_test.iloc[index]
            image = plt.imread(image_path)

            plt.subplot(3, 3, subplot_index)
            plt.imshow(image)
            np.set_printoptions(precision=5, suppress=True)
            plt.title(f"Predicted: {predicted_value/mean_value}\nActual: {'{:.5g}'.format(actual_value/mean_value)}")
            plt.axis('off')

            subplot_index += 1

    if len(unique_combinations) == 9:
        break

plt.suptitle("Bottom 9 Images with Lowest Predicted Values (Unique edgeUID and image_id)", fontsize=16)
plt.tight_layout()
plt.show()


# %%
# Get the indices of the top 9 highest predictions
top_indices = np.argsort(model_predictions.flatten())[::-1][:9]

# Plot the top 9 images with highest predicted values
plt.figure(figsize=(12, 8))
for i, index in enumerate(top_indices):
    image_path = X_test.iloc[index]
    predicted_value = model_predictions[index]
    actual_value = y_test.iloc[index]
    image = plt.imread(image_path)
    plt.subplot(3, 3, i+1)
    plt.imshow(image)
    np.set_printoptions(precision=5, suppress=True)
    plt.title(f"Predicted: {predicted_value/mean_value}\nActual: {'{:.5g}'.format(actual_value/mean_value)}")
    plt.axis('off')
plt.suptitle("Top 9 Images with Highest Predicted Values", fontsize=16)
plt.tight_layout()
plt.show()


# %%
# Get the indices of the bottom 9 lowest predictions
bottom_indices = np.argsort(model_predictions.flatten())[:9]

# Plot the bottom 9 images with lowest predicted values
plt.figure(figsize=(12, 8))
for i, index in enumerate(bottom_indices):
    image_path = X_test.iloc[index]
    predicted_value = model_predictions[index]
    actual_value = y_test.iloc[index]
    image = plt.imread(image_path)
    plt.subplot(3, 3, i+1)
    plt.imshow(image)
    np.set_printoptions(precision=5, suppress=True)
    plt.title(f"Predicted: {predicted_value/mean_value}\nActual: {'{:.5g}'.format(actual_value/mean_value)}")
    plt.axis('off')
plt.suptitle("Bottom 9 Images with Lowest Predicted Values", fontsize=16)
plt.tight_layout()
plt.show()


# %%
# Get the indices of the top 9 highest actual values
top_indices = np.argsort(y_test)[-9:]

# Plot the top 9 images with highest actual values and their predicted values
plt.figure(figsize=(12, 8))
for i, index in enumerate(top_indices):
    image_path = X_test.iloc[index]
    actual_value = y_test.iloc[index]
    predicted_value = model_predictions[index]
    image = plt.imread(image_path)
    plt.subplot(3, 3, i+1)
    plt.imshow(image)
    np.set_printoptions(precision=5, suppress=True)
    plt.title(f"Actual: {'{:.5g}'.format(actual_value/mean_value)}\nPredicted: {predicted_value/mean_value}")
    plt.axis('off')
plt.suptitle("Top 9 Images with Highest Actual Values and Predicted Values", fontsize=16)
plt.tight_layout()
plt.show()


# %%
# Get the indices of the bottom 9 lowest actual values
bottom_indices = np.argsort(y_test)[:9]

# Plot the bottom 9 images with lowest actual values and their predicted values
plt.figure(figsize=(12, 8))
for i, index in enumerate(bottom_indices):
    image_path = X_test.iloc[index]
    actual_value = y_test.iloc[index]
    predicted_value = model_predictions[index]
    image = plt.imread(image_path)
    plt.subplot(3, 3, i+1)
    plt.imshow(image)
    np.set_printoptions(precision=5, suppress=True)
    plt.title(f"Actual: {'{:.5g}'.format(actual_value/mean_value)}\nPredicted: {predicted_value/mean_value}")
    plt.axis('off')
plt.suptitle("Bottom 9 Images with Lowest Actual Values and Predicted Values", fontsize=16)
plt.tight_layout()
plt.show()


# %%
print(train_image_features)

# %%
a=train_image_features[87]
print(a.max())
print(a.min())
np.var(a)

# %%
print(np.shape(test_image_features))

from sklearn.cluster import KMeans

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assume matrix has shape of (XXX, 62720)
#scaler = StandardScaler().set_output(transform="pandas")
#test_image_features_scaled = scaler.fit_transform(test_image_features)

matrix = np.array(test_image_features)

# Apply PCA to calculate variance explained
pca = PCA()
matrix_pca = pca.fit_transform(matrix)

# Get the cumulative sum of explained variance ratios
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Find the number of components needed for 95% variance explained
n_components = np.argmax(cumulative_variance >= 0.95) + 1

print(n_components)

# %%
# Create a bar plot of the explained variance ratios
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.axvline(x=n_components, color='red', linestyle='--', label=f'{n_components} components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained by Principal Components')
plt.legend()
plt.grid(True)
plt.show()

# %%
print(np.shape(matrix_pca))

# %%
# Reduce the original PCA matrix to the components that explain 95% of the variance
matrix_pca = matrix_pca[:, :n_components+1]

print(np.shape(matrix_pca))

# %%
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

#Cluster number optimization
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)
 
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(matrix_pca)
    kmeanModel.fit(matrix_pca)
 
    distortions.append(sum(np.min(cdist(matrix_pca, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / matrix_pca.shape[0])
    inertias.append(kmeanModel.inertia_)
 
    mapping1[k] = sum(np.min(cdist(matrix_pca, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / matrix_pca.shape[0]
    mapping2[k] = kmeanModel.inertia_

# %%
plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

# %%
# Perform K-means clustering on the PCA-transformed data
k = 4  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(matrix_pca)

# Get the coordinates of the cluster centroids
centroids = kmeans.cluster_centers_

# Create a 3D scatter plot for visualization
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot each point with its corresponding cluster color
for i in range(len(matrix_pca)):
    if clusters[i] == 0:
        ax.scatter(matrix_pca[i, 0], matrix_pca[i, 1], matrix_pca[i, 2], color='red', marker='.')
    elif clusters[i] == 1:
        ax.scatter(matrix_pca[i, 0], matrix_pca[i, 1], matrix_pca[i, 2], color='blue', marker='.')
    elif clusters[i] == 2:
        ax.scatter(matrix_pca[i, 0], matrix_pca[i, 1], matrix_pca[i, 2], color='green', marker='.')
    elif clusters[i] == 3:
        ax.scatter(matrix_pca[i, 0], matrix_pca[i, 1], matrix_pca[i, 2], color='orange', marker='.')

# Plot the cluster centroids
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], centroids[:, 3], color='black', marker='x', s=100, label='Centroids')

ax.set_title('K-means Clustering with PCA')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend()
plt.show()

# %%
#The image indexes in the specific clusters

cluster1_images = []
cluster2_images = []
cluster3_images = []
cluster4_images = []
for i in range(len(clusters)):
    if clusters[i] == 0:
        cluster1_images.append(i)
    elif clusters[i] == 1:
        cluster2_images.append(i)
    elif clusters[i] == 2:
        cluster3_images.append(i)
    elif clusters[i] == 3:
        cluster4_images.append(i)

#The image paths for cluster 1
cluster1_imagepath = []
for i in cluster1_images:
#    if i in X_train:
    cluster1_imagepath.append(X_test.iloc[i])

#The image paths for cluster 2
cluster2_imagepath = []
for i in cluster2_images:
#    if i in X_train:
    cluster2_imagepath.append(X_test.iloc[i])

#The image paths for cluster 3
cluster3_imagepath = []
for i in cluster3_images:
#    if i in X_train:
    cluster3_imagepath.append(X_test.iloc[i])
    

#The image paths for cluster 4
cluster4_imagepath = []
for i in cluster4_images:
#    if i in X_train:
    cluster4_imagepath.append(X_test.iloc[i])

print(len(cluster1_imagepath), len(cluster2_imagepath), len(cluster3_imagepath), len(cluster4_imagepath))


# %% [markdown]
# img = plt.imread(cluster1_imagepath[1])
# plt.imshow(img)

# %%
print(y_test)

# %%
# Load the CSV file into a DataFrame
df = pd.read_csv('predictions2.csv')

# %%
print(df)

# %%
# Convert y-test and y-predict to numpy arrays for easier manipulation
y_test = np.array(y_test)
y_predict = np.array(df['predictions'].values) # Replace the ellipsis with the complete y-predict values

# Extract the corresponding values from y-test and y-predict using the indices
y_test_subset = y_test[cluster1_images]
y_predict_subset = y_predict[cluster1_images]

# Calculate the MAPE using the scikit-learn function
mape = mean_absolute_percentage_error(y_test_subset, y_predict_subset)

print("MAPE:", mape)
print("mean:", np.mean(y_test_subset))
print("mean pred:", np.mean(y_predict_subset))

# %%
# Convert y-test and y-predict to numpy arrays for easier manipulation
y_test = np.array(y_test)
y_predict = np.array(df['predictions'].values) # Replace the ellipsis with the complete y-predict values

# Extract the corresponding values from y-test and y-predict using the indices
y_test_subset = y_test[cluster2_images]
y_predict_subset = y_predict[cluster2_images]

# Calculate the MAPE using the scikit-learn function
mape = mean_absolute_percentage_error(y_test_subset, y_predict_subset)

print("MAPE:", mape)
print("mean:", np.mean(y_test_subset))
print("mean pred:", np.mean(y_predict_subset))

# %%
# Convert y-test and y-predict to numpy arrays for easier manipulation
y_test = np.array(y_test)
y_predict = np.array(df['predictions'].values) # Replace the ellipsis with the complete y-predict values

# Extract the corresponding values from y-test and y-predict using the indices
y_test_subset = y_test[cluster3_images]
y_predict_subset = y_predict[cluster3_images]

# Calculate the MAPE using the scikit-learn function
mape = mean_absolute_percentage_error(y_test_subset, y_predict_subset)

print("MAPE:", mape)
print("mean:", np.mean(y_test_subset))
print("mean pred:", np.mean(y_predict_subset))

# %%
# Convert y-test and y-predict to numpy arrays for easier manipulation
y_test = np.array(y_test)
y_predict = np.array(df['predictions'].values) # Replace the ellipsis with the complete y-predict values

# Extract the corresponding values from y-test and y-predict using the indices
y_test_subset = y_test[cluster4_images]
y_predict_subset = y_predict[cluster4_images]

# Calculate the MAPE using the scikit-learn function
mape = mean_absolute_percentage_error(y_test_subset, y_predict_subset)

print("MAPE:", mape)
print("mean:", np.mean(y_test_subset))
print("mean pred:", np.mean(y_predict_subset))

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# Randomly select 9 image paths from cluster2_imagepath
random_image_paths = random.sample(cluster1_imagepath, 6)

# Plot the images
fig, axes = plt.subplots(2, 3, figsize=(10, 10))
axes = axes.flatten()

for i, image_path in enumerate(random_image_paths):
    img = mpimg.imread(image_path)
    axes[i].imshow(img)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# Randomly select 9 image paths from cluster2_imagepath
random_image_paths = random.sample(cluster2_imagepath, 6)
# Plot the images
fig, axes = plt.subplots(2, 3, figsize=(10, 10))
axes = axes.flatten()

for i, image_path in enumerate(random_image_paths):
    img = mpimg.imread(image_path)
    axes[i].imshow(img)
    axes[i].axis('off')

plt.tight_layout()
plt.show()



# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# Randomly select 9 image paths from cluster2_imagepath
random_image_paths = cluster3_imagepath[10:16]

# Plot the images
fig, axes = plt.subplots(2, 3, figsize=(10, 10))
axes = axes.flatten()

for i, image_path in enumerate(random_image_paths):
    img = mpimg.imread(image_path)
    axes[i].imshow(img)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# Randomly select 9 image paths from cluster2_imagepath
random_image_paths = random.sample(cluster4_imagepath, 6)

# Plot the images
fig, axes = plt.subplots(2, 3, figsize=(10, 10))
axes = axes.flatten()

for i, image_path in enumerate(random_image_paths):
    img = mpimg.imread(image_path)
    axes[i].imshow(img)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# %%
y_tests = y_test/np.mean(y_test)
y_predict = np.array(df['predictions'].values) # Replace the ellipsis with the complete y-predict values

y_mean1 = np.mean(y_tests[cluster1_images])
y_mean2 = np.mean(y_tests[cluster2_images])
y_mean3 = np.mean(y_tests[cluster3_images])
y_mean4 = np.mean(y_tests[cluster4_images])

print(y_mean1)
print(y_mean2)
print(y_mean3)
print(y_mean4)

# %%
from scipy.stats import levene

# Perform Levene's test for homoscedasticity
levene_statistic, p_value = levene(y_tests[cluster1_images], y_tests[cluster2_images], y_tests[cluster3_images], y_tests[cluster4_images])

print("Levene's statistic:", levene_statistic)
print("p-value:", p_value)

# %%
from scipy import stats
import numpy as np

# Perform Wilcoxon rank-sum test
u_statistic, p_value = stats.mannwhitneyu(y_tests[cluster1_images], y_tests[cluster4_images])

print("Mann-Whitney U statistic:", u_statistic)
print("p-value:", p_value)

# %%



