import pickle
import os
from tqdm import tqdm

# Set the data directory path.
data_directory = 'data/'  # Replace with the actual data directory path.
dump_directory = '../XGaussian/Xray_data/'

# List all .pickle files.
pickle_files = [f for f in os.listdir(data_directory) if f.endswith('.pickle')]

# Process each file.
for file_name in tqdm(pickle_files):
    path_to_file = os.path.join(data_directory, file_name)
    path_to_dump = os.path.join(dump_directory, file_name)
    
    # Load each pickle file and resave it with protocol version 4.
    with open(path_to_file, 'rb') as handle:
        data = pickle.load(handle)
    
    with open(path_to_dump, 'wb') as handle:
        pickle.dump(data, handle, protocol=4)

print("All files have been processed.")

