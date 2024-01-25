from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best.pt")

# Open the video file
video_path = "road.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a video writer object
output_path = "Speed_tracked_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks and calculate speed
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 30 tracks for 30 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                annotated_frame,
                [points],
                isClosed=False,
                color=(255, 0, 0),
                thickness=3,
            )

            # Calculate speed
            if len(track) >= 2:
                dx = track[-1][0] - track[0][0]
                dy = track[-1][1] - track[0][1]
                distance_pixels = np.sqrt(dx ** 2 + dy ** 2)
                distance_meters = distance_pixels * 0.05  # Adjust scale if needed
                time_seconds = len(track) / fps
                speed = (distance_meters / 1000) / (time_seconds / 3600)  # Convert to km/h

                # Display speed on the frame
                label = f"Speed: {speed:.2f} km/h"
                cv2.putText(
                    annotated_frame,
                    label,
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        # Write the annotated frame to the output video
        writer.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object, writer, and close the display window
cap.release()
writer.release()
cv2.destroyAllWindows()