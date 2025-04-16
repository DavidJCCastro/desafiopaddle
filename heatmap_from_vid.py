import os
import cv2
import argparse
import numpy as np
import seaborn as sns
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
from norfair import Detection, Tracker, Video, draw_tracked_objects

def undistort_video(input_path, output_path, calibration_images=None):
    """
    Remove lens distortion from a video file
    
    Parameters:
    - input_path: Path to the input video file
    - output_path: Path to save the undistorted video
    - calibration_images: Optional list of paths to chessboard images for calibration
                         If None, uses default estimation values
    """
    # Use estimated values for a typical wide-angle camera
    # These are default values, adjust based on your specific camera if known
    print("Using default camera parameters (no calibration images provided)")
    # Camera matrix (focal length and optical centers)
    mtx = np.array([
        [1000, 0, 960],  # fx, 0, cx
        [0, 1000, 540],  # 0, fy, cy
        [0, 0, 1]
    ])
    # Distortion coefficients [k1, k2, p1, p2, k3]
    dist = np.array([[-0.3, 0.1, 0, 0, -0.02]])

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Couldn't open video file {input_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate optimal camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video with {frame_count} frames...")
    frame_idx = 0
    
    # Process each frame
    with tqdm(total=frame_count) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Undistort the frame
            dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            
            # Crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            
            # Resize back to original dimensions if needed
            if dst.shape[1] != width or dst.shape[0] != height:
                dst = cv2.resize(dst, (width, height))
            
            # Write the undistorted frame
            out.write(dst)
            
            frame_idx += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Undistorted video saved to {output_path}")
    return True
    
def detect_court_lines_for_alignment(frame, save_dir):
    """
    Detect main court lines just for alignment purposes
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate to connect edge components
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Save intermediate results for debugging
    cv2.imwrite(os.path.join(save_dir, "edges_detected.jpg"), edges)
    
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(
        dilated, rho=1, theta=np.pi/180, 
        threshold=100, minLineLength=100, maxLineGap=10
    )
    
    if lines is None or len(lines) < 3:
        print("Not enough lines detected, try lower thresholds")
        return None
    
    # Find horizontal-ish lines (court boundaries)
    h_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate angle and length
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Group based on angle (horizontal-ish lines)
        if angle < 30 or angle > 150:
            h_lines.append((line[0], length, angle))
    
    # Sort by length to get the most prominent lines
    h_lines.sort(key=lambda x: x[1], reverse=True)
    
    # Get the longest horizontal line for angle calculation
    if h_lines:
        main_line = h_lines[0][0]
        x1, y1, x2, y2 = main_line
        
        # Calculate angle of rotation needed
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if angle > 90:
            angle = angle - 180
        
        # Visualization for debugging
        line_frame = frame.copy()
        cv2.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(line_frame, f"Angle: {angle:.2f} degrees", (50, 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(save_dir,"main_alignment_line.jpg"), line_frame)
        
        return angle
    else:
        print("No suitable horizontal lines found")
        return None
        
def align_and_center_video(input_path, output_path, save_dir):
    """
    Align the court to be level and center it in the frame
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Couldn't open video file {input_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")
    
    # Read first frame for court detection
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the first frame")
        return False
    
    # Try to detect court angle
    print("Detecting court alignment...")
    angle = detect_court_lines_for_alignment(first_frame, save_dir)
    
    if angle is None:
        print("Could not detect court angle automatically.")
        # Default to no rotation
        angle = 0
    
    print(f"Detected rotation angle: {angle:.2f} degrees")
    
    # Calculate transformation matrix for rotation
    # Get rotation matrix for the angle
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions after rotation to avoid cropping
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)
    
    # Adjust the rotation matrix to take into account the new dimensions
    rotation_matrix[0, 2] += (new_width - width) // 2
    rotation_matrix[1, 2] += (new_height - height) // 2
    
    # Apply rotation to the first frame for verification
    rotated_first_frame = cv2.warpAffine(first_frame, rotation_matrix, (new_width, new_height))
    cv2.imwrite(os.path.join(save_dir,"rotated_first_frame.jpg"), rotated_first_frame)
    
    # Create video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    # Reset video capture to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print(f"Processing video with {frame_count} frames...")
    
    # Process each frame
    with tqdm(total=frame_count) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply rotation
            rotated_frame = cv2.warpAffine(frame, rotation_matrix, (new_width, new_height))
            
            # Write rotated frame
            out.write(rotated_frame)
            
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Aligned video saved to {output_path}")
    return True

def click_event(event, x, y, flags, param):
    court_points, frame = param
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked: ({x}, {y})")
        court_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Court Corners", frame)
        
def manually_grab_corners(vid_path):
    # Load a sample frame from your video
    FRAME_NUM = 1
    court_points = []
    
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUM)
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        exit()
    
    # The order is 1. Top-left, 2. Top-right, 3. Bottom-right, 4. Bottom-left
    cv2.imshow("Select Court Corners", frame)
    cv2.setMouseCallback("Select Court Corners", click_event, (court_points, frame))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    np.save("court_src_points.npy", np.array(court_points, dtype=np.float32))
    
def project_to_court(points, H):
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    court_points = cv2.perspectiveTransform(points, H)
    return court_points.reshape(-1, 2)
    
def track_players(video_path, model, tmp_dir, H, CONFIDENCE_THRESHOLD = 0.3):
    # Initialize tracker
    tracker = Tracker(distance_function="euclidean", distance_threshold=30)
    
    # Open video input/output
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(tmp_dir, "tracked_vid.mp4"), fourcc, fps, (width, height))

    all_projected_points = []
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
    
        # Run YOLO inference^
        results = model(frame)[0]
        detections = []
        projected_positions = []
        
        for det in results.boxes.data:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) != 0 or conf < CONFIDENCE_THRESHOLD:
                continue  # Only keep 'person' class
    
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            detections.append(Detection(np.array([cx.item(), cy.item()])))
            
            # You have cx, cy (center of player bbox)
            projected = project_to_court([(cx.item(), cy.item())], H)[0]
            projected_positions.append(projected)
    
        all_projected_points.append(projected_positions)
    
        # Update tracker
        tracked_objects = tracker.update(detections=detections)
    
        # Draw tracked objects
        draw_tracked_objects(frame, tracked_objects)
    
        # Write frame
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return all_projected_points
    
def create_heatmap(target_dir, positions):

    # Flatten into a single list of (x, y)
    flat_positions = [pt for frame in positions for pt in frame]
    flat_positions = np.array(flat_positions)
    
    # Separate x and y for heatmap
    x = flat_positions[:, 0]
    y = flat_positions[:, 1]
    
    # Create heatmap using 2D histogram
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[80, 40], range=[[0, 800], [0, 400]])
    # Plot heatmap
    plt.figure(figsize=(10, 5))
    sns.heatmap(heatmap, cmap="Blues", cbar=True)
    plt.title("Player Heatmap on Court")
    plt.xlabel("Court Width")
    plt.ylabel("Court Height")
    plt.savefig(target_dir)
    #plt.show()
    
def generate_heatmap_from_vid(video_path, debugging = True, manual = False):
    tmp_path = "tmp"
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

    # Remove video distortion
    undist_vid_path = os.path.join(tmp_path, "undistorted.mp4")
    undistort_video(video_path, undist_vid_path)

    # Rotate and center the video to the field
    rotated_vid_path = os.path.join(tmp_path, "rotated.mp4")
    align_and_center_video(undist_vid_path, rotated_vid_path, tmp_path)

    # The pixels of the court corners are saved to this file
    # If it doesn't exist, manually select them
    court_points_path = "court_src_points.npy"
    if not os.path.exists(court_points_path) or manual:
        manually_grab_corners(rotated_vid_path)
        
    court_src_points = np.load(court_points_path)
    # Standard paddle court size mapped to pixels
    court_width = 800
    court_height = 400
    
    court_dst_points = np.array([
        [0, 0],                      # Top-left
        [court_width, 0],           # Top-right
        [court_width, court_height],# Bottom-right
        [0, court_height]           # Bottom-left
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(court_src_points, court_dst_points)
    
    # ------------ CONFIGURATION ------------
    YOLO_MODEL = "yolov8n.pt"  # use 'yolov8s.pt' for better accuracy
    CONFIDENCE_THRESHOLD = 0.3
    # ---------------------------------------

    model = YOLO(YOLO_MODEL)

    # Run the player tracking function to get the players' positions
    proj_points = track_players(rotated_vid_path, model, tmp_path, H)

    create_heatmap("heatmap.png", proj_points)
    
    if not debugging and os.path.exists(tmp_path):
        for filename in os.listdir(tmp_path):
            file_path = os.path.join(tmp_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(path)    
        
def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Generate a court heatmap from video"
    )
    
    # Add arguments
    parser.add_argument("input_vid", nargs="?", default=None, 
                        help="Video of a paddel game")
    parser.add_argument("debugging", nargs="?", default="True", 
                        help="Determines if temporary files are deleted or kept for debugging")
    
    # Add boolean flags
    parser.add_argument("--manual", "-m", action="store_true", 
                        help="Manually choose the court corners")   
    # Parse arguments
    args = parser.parse_args()
    
    # Generate the heatmap
    generate_heatmap_from_vid(
        video_path=args.input_vid,
        debugging=args.debugging,
        manual=args.manual
    )

if __name__ == "__main__":
    main()
