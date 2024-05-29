# %%
import os
import os.path
from PIL import Image


# %%
import os
from PIL import Image, UnidentifiedImageError
from io import BytesIO

folder_path = '/home/s214626/billeder/'

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if the file is an image
    try:
        with open(file_path, 'rb') as f:
            # Attempt to open the file as an image
            img = Image.open(BytesIO(f.read()))
            img.verify()  # Verify the image file

    except (IOError, SyntaxError, UnidentifiedImageError):
        # If the file raises an error, it is not a valid image file
        # Delete the file
        os.remove(file_path)
        print(f"Deleted file: {file_path}")


# %%
source_folder = '/home/s214626/billeder'
destination_folder = '/home/s214626/unknown_images'

for file in os.listdir(source_folder):
    source_path = os.path.join(source_folder, file)

    try:
        img = Image.open(source_path)
        img = img.resize((224, 224))
        img.save(source_path)

    except OSError:
        try:
            # Move the unidentified image to the destination folder
            destination_path = os.path.join(destination_folder, file)
            os.rename(source_path, destination_path)
        except Exception as e:
            print(f"Failed to move {file}: {str(e)}")

# %%
im = Image.open('/home/s214626/billeder/591946748969801.jpg')

#show image
im.show()

# %%
lst = os.listdir('/home/s214626/billeder') # your directory path
print(len(lst))

# %%



