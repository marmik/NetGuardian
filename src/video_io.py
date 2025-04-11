import cv2
import time

def handle_video_io(input_source, output_path, display=False):
    """
    Handle real-time video input/output using OpenCV.
    Args:
        input_source (str or int): Path to the input video file or camera index.
        output_path (str): Path to save the output video file.
        display (bool): Whether to display the video in a window.
    """
    cap = cv2.VideoCapture(input_source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame (e.g., object detection, tracking, etc.)
        # (Implement processing code here)

        out.write(frame)

        if display:
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    if display:
        cv2.destroyAllWindows()

def log_fps(input_source):
    """
    Log FPS and other metrics.
    Args:
        input_source (str or int): Path to the input video file or camera index.
    """
    cap = cv2.VideoCapture(input_source)
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time

    print(f"Total frames: {frame_count}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"FPS: {fps:.2f}")

    cap.release()

def support_rtsp_stream(rtsp_url, output_path, display=False):
    """
    Support RTSP streams.
    Args:
        rtsp_url (str): URL of the RTSP stream.
        output_path (str): Path to save the output video file.
        display (bool): Whether to display the video in a window.
    """
    handle_video_io(rtsp_url, output_path, display)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Real-time video input/output using OpenCV')
    parser.add_argument('--input-source', type=str, default='0', help='Path to the input video file or camera index')
    parser.add_argument('--output-path', type=str, default='output.mp4', help='Path to save the output video file')
    parser.add_argument('--display', action='store_true', help='Display the video in a window')
    parser.add_argument('--log-fps', action='store_true', help='Log FPS and other metrics')
    parser.add_argument('--rtsp-url', type=str, help='URL of the RTSP stream')

    args = parser.parse_args()

    if args.log_fps:
        log_fps(args.input_source)
    elif args.rtsp_url:
        support_rtsp_stream(args.rtsp_url, args.output_path, args.display)
    else:
        handle_video_io(args.input_source, args.output_path, args.display)
