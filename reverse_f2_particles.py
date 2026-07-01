import h5py
import os
import re

def reverse_particles(input_path, output_path):
    print(f"Opening input file: {input_path}")
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    with h5py.File(input_path, 'r') as fin:
        # Find all keys that match 'Particle X'
        particle_keys = [k for k in fin.keys() if k.startswith('Particle ')]
        
        # Extract the numbers to sort them correctly
        def get_particle_num(key):
            match = re.search(r'Particle (\d+)', key)
            return int(match.group(1)) if match else 0
        
        particle_keys.sort(key=get_particle_num)
        
        print(f"Found particles: {particle_keys}")
        
        # Reverse the list of keys
        reversed_keys = particle_keys[::-1]
        
        print(f"Reversing order: {reversed_keys} -> {particle_keys}")

        with h5py.File(output_path, 'w') as fout:
            # Copy non-particle keys first
            for key in fin.keys():
                if not key.startswith('Particle '):
                    print(f"Copying {key}...")
                    fin.copy(key, fout)
            
            # Copy particles in reversed order
            for i, old_key in enumerate(reversed_keys):
                new_key = f"Particle {i + 1}"
                print(f"Copying {old_key} as {new_key}...")
                
                # Copy the group
                fin.copy(old_key, fout, name=new_key)

    print(f"Success! Reversed file saved to: {output_path}")

if __name__ == "__main__":
    folder_path = '/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/19 June 2026/'
    input_file = os.path.join(folder_path, 'TRAST long timescale F2.h5')
    output_file = os.path.join(folder_path, 'TRAST long timescale F2_reversed.h5')
    
    reverse_particles(input_file, output_file)
