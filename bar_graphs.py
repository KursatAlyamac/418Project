import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to parse the Output column
def parse_output(output):
    times = []
    
    for line in output.split('\n'):
        if not line.startswith('Finished'):
            parts = line.strip().split()
            if len(parts) == 2:
                times.append(int(parts[0]))
    
    return pd.DataFrame({'time': times})

# Read the CSV
df = pd.read_csv('results_20241210_163056.csv')

# Create lists to store processed data
all_times = []
all_tracks = []
all_versions = []

# Track names in order
tracks = ['zengarden', 'lighthouse', 'hacienda', 'snowtuxpeak', 'cornfield_crossing', 'scotland']

# Process each row
for _, row in df.iterrows():
    parsed = parse_output(row['Output'])
    times = parsed['time'].tolist()
    
    all_times.extend(times)
    all_versions.extend([row['Directory']] * len(times))
    all_tracks.extend(tracks[:len(times)])  # In case of incomplete data

# Create processed dataframe
processed_df = pd.DataFrame({
    'Version': all_versions,
    'Track': all_tracks,
    'Time': all_times,
})

# Calculate averages for each version and track
averages = processed_df.groupby(['Version', 'Track'])['Time'].mean().round(2).unstack()

# Create the bar plot
plt.figure(figsize=(15, 8))
bar_width = 0.13
index = np.arange(len(averages.index))

# Plot bars for each track
for i, track in enumerate(tracks):
    plt.bar(index + i*bar_width, averages[track], 
            bar_width, label=track)

plt.xlabel('Model Version')
plt.ylabel('Time (steps)')
plt.title('Average Completion Time by Model Version and Track')
plt.xticks(index + bar_width * 2.5, averages.index, rotation=45)
plt.legend()

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('completion_times.png', dpi=300, bbox_inches='tight')

# Print the averages
print("\nAverage Completion Times by Track:")
print(averages)

# Also save averages to CSV
averages.to_csv('completion_times.csv') 