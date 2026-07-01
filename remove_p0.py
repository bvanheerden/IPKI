import os
import json

base_dir = "/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/29 May 2026/Thylakoid power study"

for root, dirs, files in os.walk(base_dir):
    if "config.json" in files:
        file_path = os.path.join(root, "config.json")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if "p0" in data:
                print(f"Removing 'p0' from {file_path}")
                del data["p0"]
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
            else:
                print(f"'p0' not found in {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
