import cv2
from dotenv import load_dotenv
import os

load_dotenv()
# Get a VideoCapture object


RAW_VIDEO_PATH = os.getenv("RAW_VIDEO_PATH")

OUTPUT_SEGMENT_PATH = os.getenv("SEGMENT_PATH")

def extract_segment():

    input_video = cv2.VideoCapture(RAW_VIDEO_PATH)

    if not input_video.isOpened():
        print("Error in ExtractSegment: Could not open video.")
        exit()

    # get frame and duration info
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps


    # Get the frame range of the segment
    segment_duration = 20 # Segment duration in seconds
    start_time = 0
    start_frame = start_time * fps

    # We get the end frame of the segment
    # or the last frame if the segment exceeds the video length.
    end_frame = min(start_frame + (segment_duration * fps), frame_count)

    # Set the video to the segment start frame
    input_video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    width = input_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Set up codec and VideoWriter object
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(OUTPUT_SEGMENT_PATH, codec, fps,
                                    (int(width), int(height)))

    current_frame = start_frame
    while input_video.isOpened() and current_frame < end_frame:
        is_read, frame = input_video.read()
        if not is_read:
            print("Error in ExtractSegment: Could not read frame.")
            break
        
        print(f"Writing frame {current_frame + 1} / {end_frame}")
        output_video.write(frame)
        current_frame += 1

    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    extract_segment()