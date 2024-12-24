import cv2
import inspect
from ultralytics import YOLO, solutions

# Load the pre-trained YOLOv8 model
model = YOLO(r"D:\final_model_version2/best.pt")
# Open the video file (webcam feed)
cap = cv2.VideoCapture(r"C:\Users\admin\PycharmProjects\yolov8\20241111_111609.mp4")
assert cap.isOpened(), "Error reading video file"

# Get video properties: width, height, and frames per second (fps)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
print(w, h)

# Define points for a line or region of interest in the video frame
line_points = [(0, h), (w, h), (w, 0), (0, 0)]  # Line coordinates

# Specify classes to count, for example: t-shirts (class IDs 0-13)
classes_to_count = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# Initialize the video writer to save the output video
output_video_path = "D://final_model//object_counting_output2.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

# Initialize the Object Counter with visualization options and other parameters
counter = solutions.ObjectCounter(
    view_img=True,  # Display the image during processing
    reg_pts=line_points,  # Region of interest points
    names=model.names,  # Class names from the YOLO model
    draw_tracks=True,  # Draw tracking lines for objects
    line_thickness=1,  # Thickness of the lines drawn
    track_color=True,
    view_out_counts=True
)

# Process video frames in a loop
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Perform object tracking on the current frame, filtering by specified classes
    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count, conf=0.7)

    # Use the Object Counter to count objects in the frame and get the annotated image
    im0 = counter.start_counting(im0, tracks)

    # Write the processed frame to the output video
    video_writer.write(im0)

# Release the video capture and writer objects
cap.release()
video_writer.release()  # Save and release the output video file

# Close all OpenCV windows
cv2.destroyAllWindows()

print(inspect.signature(solutions.ObjectCounter.__init__))
