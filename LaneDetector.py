import os
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

RAW_VIDEO_PATH = os.getenv("RAW_VIDEO_PATH")
INPUT_SEGMENT_PATH = os.getenv("SEGMENT_PATH")
RAW_NIGHT_VIDEO_PATH = os.getenv("RAW_NIGHT_VIDEO_PATH")
CARS_XML = os.getenv("CARS_XML")

def roi_mask(input_frame):
    frame = input_frame.copy()
    height, width = frame.shape[:2]
    
    # Define a polygon that covers the bottom region of the image
    trapezoid = np.array([[
        (int(0.02 * width), height),
        (int(0.45 * width), int(0.55 * height)),
        (int(0.55 * width), int(0.55 * height)),
        (int(0.99 * width), height)
    ]], dtype=np.int32)
    
    mask = cv2.fillPoly(np.zeros_like(frame), trapezoid, 255)
    masked_frame = cv2.bitwise_and(frame, mask)
    return masked_frame

def detect_cars(input_frame, input_cascade):
    # Get rid of sides and upper part of the frame
    frame = input_frame.copy()
    height, width = frame.shape[:2]
    roi = np.array([[
        (int(0.05 * width), height),
        (int(0.05 * width), int(0.45 * height)),
        (int(0.95 * width), int(0.45 * height)),
        (int(0.95 * width), height)

    ]], dtype=np.int32)
    mask = cv2.fillPoly(np.zeros_like(frame), roi, 255)
    masked_frame = cv2.bitwise_and(frame, mask)
    # convert to grayscale
    gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2GRAY)

    cars = input_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=7)

    return cars, gray_frame


def any_overlap_with_lane(x, y, w, h, lane_mask):
    """
    Returns True if ANY part of the bounding box (x,y,w,h)
    overlaps the lane_mask (white pixels).
    """
    # Ensure we don't go out of bounds if the detection extends beyond image edges
    y1, y2 = max(0, y), min(y + h, lane_mask.shape[0])
    x1, x2 = max(0, x), min(x + w, lane_mask.shape[1])
    
    # Slice the relevant region in the mask
    sub_mask = lane_mask[y1:y2, x1:x2]
    
    # If any pixel in sub_mask is nonzero => overlap
    return cv2.countNonZero(sub_mask) > 0

def process_frame(input_frame):
    frame = input_frame.copy()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    dilated_frame = cv2.dilate(blurred_frame, kernel)
    edges = cv2.Canny(dilated_frame, 50, 100)
    masked_frame = roi_mask(edges)
    return masked_frame

def get_coordinates(input_rho, input_theta):
    a = np.cos(input_theta)
    b = np.sin(input_theta)
    x0 = a * input_rho
    y0 = b * input_rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * (a))
    return x1, y1, x2, y2


def detect_lanes(
    input_video_path=INPUT_SEGMENT_PATH, 
    output_video_path="output.mp4"
):
    # Get a VideoCapture object
    input_video = cv2.VideoCapture(input_video_path)

    # Get the video's width, height, and fps
    vid_width = int(input_video.get(3))
    vid_height = int(input_video.get(4))
    vid_fps = input_video.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object to write output
    writer = cv2.VideoWriter(
        output_video_path, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        vid_fps, 
        (vid_width, vid_height)
    )

    # Lists for lane approximation, each object is [rho, theta]
    prev_left_lanes = []
    prev_right_lanes = []

    # Lane change variables
    right_change = False
    left_change = False
    text_frames = 0
    prev_center_values = []

    # Other variables' initialization
    frame_count = 0
    read = True
    in_area = False

    car_cascade = cv2.CascadeClassifier(CARS_XML)

    while read:
        read, frame = input_video.read()
        if not read:
            break
            
        processed_frame = process_frame(frame)

        left_lines = cv2.HoughLines(
            processed_frame, 
            1, 
            np.pi / 180, 
            140, 
            max_theta=(6 * np.pi) / 18, 
            min_theta=(0 * np.pi) / 18
        )
        right_lines = cv2.HoughLines(
            processed_frame, 
            1, 
            np.pi / 180, 
            140, 
            min_theta=(11 * np.pi) / 18, 
            max_theta=(14 * np.pi) / 18
        )

        # Approximate lines if none detected
        if left_lines is None and len(prev_left_lanes) > 0:
            left_lines = np.array([[prev_left_lanes[-1][0]]])
        if right_lines is None and len(prev_right_lanes) > 0:
            right_lines = np.array([[prev_right_lanes[-1][0]]])

        # Filter lines that are too different from previous lines
        if len(prev_left_lanes) > 0 and left_lines is not None:
            close_left_lines = [
                line for line in left_lines 
                if abs(line[0][0] - prev_left_lanes[-1][0][0]) < 50 
                and abs(line[0][1] - prev_left_lanes[-1][0][1]) < 0.1
            ]
            if len(close_left_lines) > 0:
                left_lines = close_left_lines
        
        if len(prev_right_lanes) > 0 and right_lines is not None:
            close_right_lines = [
                line for line in right_lines 
                if abs(line[0][0] - prev_right_lanes[-1][0][0]) < 50
                and abs(line[0][1] - prev_right_lanes[-1][0][1]) < 0.1
            ]
            if len(close_right_lines) > 0:
                right_lines = close_right_lines
                
        # Get average rho and theta
        if left_lines is not None:
            left_rho = np.average([line[0][0] for line in left_lines])
            left_theta = np.average([line[0][1] for line in left_lines])
        else:
            left_rho = 0
            left_theta = 0

        if right_lines is not None:
            right_rho = np.average([line[0][0] for line in right_lines])
            right_theta = np.average([line[0][1] for line in right_lines])
        else:
            right_rho = 0
            right_theta = 0
        
        # Coordinates
        x1, y1, x2, y2 = get_coordinates(left_rho, left_theta)
        x3, y3, x4, y4 = get_coordinates(right_rho, right_theta)

        # Crop lines top, 620 top for day, 720 for night
        if y2 < 640:
            x2 = int(x2 + (640 - y2) * (x1 - x2) / (y1 - y2))
            y2 = 640
        if y3 < 640:
            x3 = int(x3 + (640 - y3) * (x4 - x3) / (y4 - y3))
            y3 = 640

        if y1 > 950:  # crop bottom
            x1 = int(x1 + (950 - y1) * (x2 - x1) / (y2 - y1))
            y1 = 950
        if y4 > 950:  
            x4 = int(x4 + (950 - y4) * (x3 - x4) / (y3 - y4))
            y4 = 950
            
        center = (x2 + x3) / 2

        # Lane-change detection
        if len(prev_left_lanes) > 0:
            left_rho_avg = np.average([line[0][0] for line in prev_left_lanes[-25:]])
        else: 
            left_rho_avg = 0

        if len(prev_right_lanes) > 0:
            right_rho_avg = np.average([line[0][0] for line in prev_right_lanes[-25:]])
        else:
            right_rho_avg = 0
        
        if len(prev_center_values) > 0:
            center_avg = np.average(prev_center_values[-10:])
        else:
            center_avg = 0

        left_diff = abs(left_rho - left_rho_avg)
        right_diff = abs(right_rho - right_rho_avg)

        if not left_change and not right_change and left_diff > 50 and right_diff > 50:
            left_change = center - 15 < center_avg
            right_change = center - 20 >= center_avg
            text_frames = 0
   
        prev_left_lanes.append([[left_rho, left_theta]])
        prev_right_lanes.append([[right_rho, right_theta]])
        prev_center_values.append(center)

        if left_change or right_change:
            if text_frames < 40:
                cv2.putText(
                    frame, 
                    "Change to left lane" if left_change else "Change to right lane", 
                    (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    2, 
                    (0, 255, 0), 
                    4
                )
                text_frames += 1
            else:
                left_change = False
                right_change = False
                text_frames = 0

        # Detect cars

        cars, debug_frame = detect_cars(frame, car_cascade)

        lane_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        if left_lines is not None and right_lines is not None:
            # Draw the lines
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
            cv2.line(frame, (x3, y3), (x4, y4), (255, 0, 0), 5)

            # Define the polygon for lane area
            lane_points = np.array([[
                (x1, y1), 
                (x2, y2), 
                (x3, y3), 
                (x4, y4)
            ]], dtype=np.int32)

            # Fill that polygon on lane_mask
            cv2.fillPoly(lane_mask, lane_points, 255)

            # Optionally color the lane area (green) with some transparency
            color_mask = np.zeros_like(frame)
            color_mask[:, :, 1] = lane_mask  # fill the green channel
            alpha = 0.3
            frame = cv2.addWeighted(frame, 1, color_mask, alpha, 0)

            # Calculate & print lane area in pixels
            lane_area = cv2.countNonZero(lane_mask)
            cv2.putText(
                frame, 
                f"Lane area: {lane_area} px", 
                (50, 270), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 255), 
                2
            )
        for (x, y, w, h) in cars:
            if any_overlap_with_lane(x, y, w, h, lane_mask):
                color = (0, 0, 255)  # Red if overlap with lane mask
            else:
                color = (0, 255, 0)  # Green if no overlap

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # Write frame to output
        writer.write(frame)

        # Debug / visualization
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        print(f"frame: {frame_count} processed successfully.")

    input_video.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_lanes(RAW_VIDEO_PATH, "output.mp4")
