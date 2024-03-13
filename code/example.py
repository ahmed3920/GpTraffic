#
# import cv2
#
# from Airlight import Airlight
# from BoundCon import BoundCon
# from CalTransmission import CalTransmission
# from removeHaze import removeHaze
#
# if __name__ == '__main__':
#     HazeImg = cv2.imread('664.jpg')
#
#
#
#     # Estimate Airlight
#     windowSze = 15
#     AirlightMethod = 'fast'
#     A = Airlight(HazeImg, AirlightMethod, windowSze)
#
#     # Calculate Boundary Constraints
#     windowSze = 3
#     C0 = 20         # Default value = 20
#     C1 = 300        # Default value = 300
#     Transmission = BoundCon(HazeImg, A, C0, C1, windowSze)
#
#     # Refine estimate of transmission
#     regularize_lambda = 1  # Default value = 1  --> Regularization parameter, the more this  value, the closer to the original patch wise transmission
#     sigma = 0.5
#     Transmission = CalTransmission(HazeImg, Transmission, regularize_lambda, sigma)     # Using contextual information
#
#     # Perform DeHazing
#     HazeCorrectedImg = removeHaze(HazeImg, Transmission, A, 0.85)
#
#
#     cv2.imwrite('668.jpeg', HazeCorrectedImg)
import cv2
from Airlight import Airlight  # Assuming these modules support videos or can be adapted
from BoundCon import BoundCon
from CalTransmission import CalTransmission
from removeHaze import removeHaze

def main():
    # Video input path (replace with your video filename and path)
    video_path = "test.mp4"

    # Open video capture
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video!")
        return

    # Define output format (adjust as needed)
    output_format = ".jpg"  # Save dehazed frames as images

    # Loop through each frame
    while True:
        ret, frame = cap.read()

        # Check if frame is read successfully
        if not ret:
            break

        # Dehazing process (same logic as before)
        windowSze = 15
        AirlightMethod = 'fast'
        A = Airlight(frame, AirlightMethod, windowSze)

        windowSze = 3
        C0 = 20
        C1 = 300
        Transmission = BoundCon(frame, A, C0, C1, windowSze)

        regularize_lambda = 1
        sigma = 0.5
        Transmission = CalTransmission(frame, Transmission, regularize_lambda, sigma)

        HazeCorrectedImg = removeHaze(frame, Transmission, A, 0.85)

        # Save dehazed frame (or write to output video if desired)
        output_name = f"dehazed_frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.{output_format}"
        cv2.imwrite(output_name, HazeCorrectedImg)

        # Display frame for visual verification (optional)
        # cv2.imshow('Dehazed Frame', HazeCorrectedImg)
        # cv2.waitKey(1)  # Adjust wait time for slower display

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
