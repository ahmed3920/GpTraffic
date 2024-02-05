import json
from pathlib import Path


def _get_lane_destination(lane_index):
    # Helper method to determine the destination based on the lane index
    if lane_index in [0, 1, 3, 4, 6,7, 9, 10]:
        return 'multi_direction'
    elif lane_index == 2:
        return 'south'
    elif lane_index == 5:
        return 'north'
    elif lane_index == 8:
        return 'west'
    elif lane_index == 11:
        return 'east'
    else:
        return 1  # Default value if the lane index is unexpected


class ResultsSaver:
    def __init__(self, save_dir):
        # Constructor to initialize the ResultsSaver object with a save directory
        self.save_dir = Path(save_dir)

    def save_results(self, frame, direction, total_counts, lane_counts, object_counts):
        # Method to save results for a specific frame and direction
        lane_indices = { "East": [0, 1, 2], "West": [3, 4, 5], "South": [6, 7, 8],"North": [9, 10, 11]}

        # Create a dictionary to store results in JSON format
        json_data = {
            'Total number of vehicles': total_counts,
            'lanes': {},
        }

        # Iterate over lane indices for the given direction
        for i in lane_indices.get(direction, []):
            lane_key = f'Lane{i + 1}'

            # Populate lane-specific data in the JSON structure
            json_data['lanes'][lane_key] = {
                'Total': lane_counts.get(i, 0),
                'ObjectCounts': {label: object_counts[i][label] for label in ["car", "bus", "truck", "cycle"]},

                'destination': _get_lane_destination(i)
            }

        # Define the JSON file name based on frame and direction
        json_filename = f'results_{frame}_{direction}.json'
        json_filepath = self.save_dir / json_filename

        # Write the JSON data to the file
        with open(json_filepath, 'w') as json_file:
            json.dump(json_data, json_file, indent=2)

        # Return the file path as a string
        return str(json_filepath)

    def save_master_results(self, frame, total_counts_direction, total_counts, all_object_counts):
        # Method to save master results combining data from different directions
        master_json_data = {
            'Total number of vehicles': total_counts,
            'lanes': {},
            'ObjectCounts': all_object_counts
        }

        # Iterate over total counts for each direction
        for direction, direction_total in total_counts_direction.items():
            # Read individual direction results from the corresponding file
            direction_json_file = f'results_{frame}_{direction}.json'
            with open(self.save_dir / direction_json_file, 'r') as json_file:
                direction_data = json.load(json_file)

            # Add lane-specific data to the master JSON structure
            master_json_data['lanes'][direction] = direction_data['lanes']

        # Define the master JSON file name based on the frame
        master_json_filename = f'master_results_{frame}.json'
        master_json_filepath = self.save_dir / master_json_filename

        # Write the master JSON data to the file
        with open(master_json_filepath, 'w') as master_json_file:
            json.dump(master_json_data, master_json_file, indent=4)

        # Return the file path as a string
        return str(master_json_filepath)
