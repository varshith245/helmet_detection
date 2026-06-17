# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 19:37:31 2025

@author: hp
"""
import os
import cv2
import time
import argparse
from datetime import datetime
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Real-time webcam inference using YOLOv8")
    
    default_weights = "model/yolov8/best.pt" if os.path.exists("model/yolov8/best.pt") else "best.pt"
    parser.add_argument("--weights", type=str, default=default_weights, help="Path to model weights")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0 = default built-in, 1 or 2 for external)")
    parser.add_argument("--output-dir", type=str, default="video_feed", help="Directory to save output video")
    parser.add_argument("--no-save", action="store_true", help="Disable saving the video feed to file")
    parser.add_argument("--duration", type=int, default=15, help="Duration to run in seconds (set to 0 for infinite)")
    
    args = parser.parse_args()

    # Load trained YOLO model
    print(f"Loading model from: {args.weights}")
    try:
        model = YOLO(args.weights)
    except Exception as e:
        print(f"Could not load YOLO model from {args.weights}. Error: {e}")
        return

    # Try opening the requested camera, fallback to index 0 if it fails
    camera_idx = args.camera
    print(f"Attempting to open camera with index {camera_idx}...")
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    
    if not cap.isOpened() and camera_idx != 0:
        print(f"Failed to open camera {camera_idx}. Falling back to default camera (index 0)...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
    if not cap.isOpened():
        print("Error: Could not open any camera.")
        return

    # Get FPS from camera
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:  # Sometimes returns 0 or NaN
        fps = 10.0
    print("Camera FPS:", fps)

    # Create output folder
    save_output = not args.no_save
    if save_output:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"webcam_output_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID is widely supported
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Saving output video to: {output_path}")

    # Run loop
    start_time = time.time()
    print("Recording started. Press 'q' to quit.")
    time.sleep(0.5)
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from camera")
            break

        # Run YOLO detection
        results = model.predict(frame, verbose=False)

        # Annotate results
        annotated_frame = results[0].plot()

        # Show live window
        cv2.imshow("YOLO Helmet Detection", annotated_frame)

        # Save video frame
        if save_output:
            out.write(annotated_frame)

        # Exit on 'q' or after timeout duration
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⏹ Exiting on key press")
            break
            
        if args.duration > 0 and (time.time() - start_time) > args.duration:
            print(f"⏹ Exiting after reaching configured duration limit of {args.duration} seconds")
            break

    # Release everything
    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
