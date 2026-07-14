import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import numpy as np

for filename in ['data_no_aa.pkl', 'data_with_aa.pkl', 'variables.pkl']:
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            print(f"\n--- {filename} ---")
            print(f"Type: {type(data)}")
            if isinstance(data, list):
                print(f"Length: {len(data)}")
                if len(data) > 0:
                    print(f"First element type: {type(data[0])}")
                    if isinstance(data[0], dict):
                        print(f"Keys: {data[0].keys()}")
            elif isinstance(data, dict):
                print(f"Keys: {data.keys()}")
            else:
                print(data)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
