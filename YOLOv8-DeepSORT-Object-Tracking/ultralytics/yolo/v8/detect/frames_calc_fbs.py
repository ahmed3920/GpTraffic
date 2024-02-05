import cv2
import os

def frames_to_video(frames_folder, output_video_path, fps=2):
    # Get the list of frame files in the folder
    frame_files = [f for f in os.listdir(frames_folder) if f.endswith('.png') or f.endswith('.jpg')]
    frame_files.sort()

    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, layers = first_frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' or 'MJPG' depending on the codec availability
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write frames to video
    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()

    print(f"Video created at: {output_video_path}")

# Example usage
frames_folder_path = 'frames_req'
output_video_path = 'new_tesing_fbs/test.mp4'
frames_to_video(frames_folder_path, output_video_path)