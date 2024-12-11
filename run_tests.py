import os
import subprocess
import glob
import csv
import time
from datetime import datetime

def find_powerups_dirs():
    # Find all directories starting with 'powerups'
    return [d for d in glob.glob('powerups*') if os.path.isdir(d)]

def run_command(directory):
    # Change to the directory and run the command
    command = 'python homework/planner.py zengarden lighthouse hacienda snowtuxpeak cornfield_crossing scotland -v'
    
    try:
        # Change directory
        os.chdir(directory)
        # Run command and capture output
        result = subprocess.run(command.split(), capture_output=True, text=True)
        # Change back to original directory
        os.chdir('..')
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # Create timestamp for CSV filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'results_{timestamp}.csv'
    
    # Find powerups directories
    powerups_dirs = find_powerups_dirs()
    
    # Prepare CSV headers
    headers = ['Directory', 'Run', 'Output']
    
    # Open CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        # For each directory
        for directory in powerups_dirs:
            print(f"Processing {directory}...")
            
            # Run 5 times
            for run in range(1, 6):
                print(f"  Run {run}/5...")
                output = run_command(directory)
                writer.writerow([directory, run, output])
                
                # Small delay between runs
                if run < 5:
                    time.sleep(1)
    
    print(f"\nResults saved to {csv_filename}")

if __name__ == "__main__":
    main() 