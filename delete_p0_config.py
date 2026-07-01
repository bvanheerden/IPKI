import os
import json

base_data_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/29 May 2026/Power study LHCII'

def delete_p0_from_configs(base_dir):
    print(f"Scanning directory: {base_dir}")
    count = 0
    for root, dirs, files in os.walk(base_dir):
        if 'config.json' in files:
            file_path = os.path.join(root, 'config.json')
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'p0' in data:
                    del data['p0']
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=4)
                    print(f"✓ Removed 'p0' from {file_path}")
                    count += 1
                else:
                    print(f"- 'p0' not found in {file_path}")
            except Exception as e:
                print(f"✗ Error processing {file_path}: {e}")
    
    print(f"\nFinished. Updated {count} files.")

if __name__ == "__main__":
    delete_p0_from_configs(base_data_dir)
