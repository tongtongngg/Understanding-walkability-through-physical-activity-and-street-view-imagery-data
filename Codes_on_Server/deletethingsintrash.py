{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8037650d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os, shutil\n",
    "folder = '/home/s214626/.local/share/Trash'\n",
    "for filename in os.listdir(folder):\n",
    "    file_path = os.path.join(folder, filename)\n",
    "    try:\n",
    "        if os.path.isfile(file_path) or os.path.islink(file_path):\n",
    "            os.unlink(file_path)\n",
    "        elif os.path.isdir(file_path):\n",
    "            shutil.rmtree(file_path)\n",
    "    except Exception as e:\n",
    "        print('Failed to delete %s. Reason: %s' % (file_path, e))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "60b785ed",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
