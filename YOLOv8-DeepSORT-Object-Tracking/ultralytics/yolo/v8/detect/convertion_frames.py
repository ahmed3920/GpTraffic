import cv2
import os

def video_to_images(video_path, output_folder, frame_interval=1):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval in frames
    interval_frames = int(fps * frame_interval)

    # Loop through the frames and save images
    for i in range(0, total_frames, interval_frames):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        # Read the frame
        ret, frame = cap.read()

        # Break the loop if the video ends
        if not ret:
            break

        # Save the frame as an image
        image_path = os.path.join(output_folder, f"frame_{i}.jpg")
        cv2.imwrite(image_path, frame)

    # Release the video capture object
    cap.release()

# Example usage
video_path = "testing.mp4"
output_folder = "frames_req"
frame_interval = int(9252/1850)

video_to_images(video_path, output_folder, frame_interval)