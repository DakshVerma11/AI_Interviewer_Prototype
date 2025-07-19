# analysis/eye_tracking.py
import cv2
import numpy as np
import os
import json
from datetime import datetime

class OptimizedEyeTracker:
    """
    Analyzes video frames for face and eye movements to detect potential cheating.
    Optimized for 24 FPS video input.
    """
    def __init__(self):
        # Initialize OpenCV classifiers for face and eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Optimized tracking parameters for 24 FPS video
        self.prev_face_center = None
        self.prev_eye_centers = None
        self.face_movement_threshold = 25  # Threshold for significant face movement (pixels)
        self.eye_movement_threshold = 18   # Threshold for significant eye movement (pixels)
        
        # Frame analysis optimization: analyze every Nth frame
        # For 24 FPS video, analyzing every 3rd frame gives an effective analysis rate of 8 FPS
        self.analysis_frame_skip = 3  
        
    def get_center(self, rect):
        """Calculates the center point of a given rectangle (x, y, w, h)."""
        x, y, w, h = rect
        return (x + w // 2, y + h // 2)
    
    def calculate_distance(self, point1, point2):
        """Calculates the Euclidean distance between two points."""
        if point1 is None or point2 is None:
            return 0
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def detect_eye_in_face(self, face_roi):
        """
        Detects eyes within a given face region of interest (ROI).
        Parameters are optimized for performance.
        """
        eyes = self.eye_cascade.detectMultiScale(
            face_roi, 
            scaleFactor=1.15,  # How much the image size is reduced at each image scale
            minNeighbors=4,    # Minimum number of neighbors each candidate rectangle should have
            minSize=(12, 12),  # Minimum possible object size. Objects smaller than that are ignored.
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return eyes
    
    def analyze_frame(self, frame):
        """
        Analyzes a single video frame for face and eye detection,
        calculates movements, and determines if the subject is looking away.
        """
        # Resize frame for faster processing if it's too large
        height, width = frame.shape[:2]
        scale_factor = 1.0
        if width > 640: # If width is greater than 640 pixels, scale it down
            scale_factor = 640 / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to grayscale for cascade classifiers
        
        # Detect faces in the grayscale frame
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(60, 60),  # Minimum face size to detect
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        analysis_result = {
            'faces_detected': len(faces),
            'face_movement': 0,
            'eye_movement': 0,
            'looking_away': False,
            'eyes_detected': 0,
            'frame_scale': scale_factor
        }
        
        if len(faces) > 0:
            # Assume the largest face is the main subject
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            
            # Adjust coordinates back to original scale if the frame was resized
            if scale_factor != 1.0:
                x = int(x / scale_factor)
                y = int(y / scale_factor)
                w = int(w / scale_factor)
                h = int(h / scale_factor)
            
            # Calculate face center and movement
            face_center = self.get_center((x, y, w, h))
            if self.prev_face_center is not None:
                face_movement = self.calculate_distance(face_center, self.prev_face_center)
                analysis_result['face_movement'] = face_movement
                
                # Flag as looking away if face movement exceeds threshold
                if face_movement > self.face_movement_threshold:
                    analysis_result['looking_away'] = True
            
            self.prev_face_center = face_center
            
            # Define region of interest for eyes within the detected face
            face_roi_y1 = max(0, int(y * scale_factor))
            face_roi_y2 = min(gray.shape[0], int((y + h) * scale_factor))
            face_roi_x1 = max(0, int(x * scale_factor))
            face_roi_x2 = min(gray.shape[1], int((x + w) * scale_factor))
            
            face_roi = gray[face_roi_y1:face_roi_y2, face_roi_x1:face_roi_x2]
            
            if face_roi.size > 0:
                eyes = self.detect_eye_in_face(face_roi)
                analysis_result['eyes_detected'] = len(eyes)
                
                # Analyze eye movement if at least two eyes are detected
                if len(eyes) >= 2:
                    # Sort eyes by x-coordinate to get a consistent order (left, right)
                    eyes = sorted(eyes, key=lambda e: e[0])
                    
                    eye_centers = []
                    for eye in eyes[:2]: # Consider only the first two detected eyes
                        ex, ey, ew, eh = eye
                        # Convert eye coordinates back to full frame coordinates
                        eye_center_x = face_roi_x1 + ex + ew//2
                        eye_center_y = face_roi_y1 + ey + eh//2
                        
                        if scale_factor != 1.0:
                            eye_center_x = int(eye_center_x / scale_factor)
                            eye_center_y = int(eye_center_y / scale_factor)
                        
                        eye_centers.append((eye_center_x, eye_center_y))
                    
                    # Calculate eye movement
                    if self.prev_eye_centers is not None and len(self.prev_eye_centers) == len(eye_centers):
                        total_eye_movement = 0
                        for i, (current, previous) in enumerate(zip(eye_centers, self.prev_eye_centers)):
                            movement = self.calculate_distance(current, previous)
                            total_eye_movement += movement
                        
                        avg_eye_movement = total_eye_movement / len(eye_centers)
                        analysis_result['eye_movement'] = avg_eye_movement
                        
                        # Flag as looking away if average eye movement exceeds threshold
                        if avg_eye_movement > self.eye_movement_threshold:
                            analysis_result['looking_away'] = True
                    
                    self.prev_eye_centers = eye_centers
                else:
                    # If eyes are not detected (or less than 2), consider it as looking away
                    analysis_result['looking_away'] = True
                    self.prev_eye_centers = None
        else:
            # If no face is detected, it's definitely looking away
            analysis_result['looking_away'] = True
            self.prev_face_center = None
            self.prev_eye_centers = None
        
        return analysis_result
    
    def analyze_video_for_cheating(self, video_path, output_dir):
        """
        Analyzes a video file for eye movement and suspicious behavior
        to detect potential cheating.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise Exception(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Tracking variables for analysis
            frame_count = 0
            looking_away_frames = 0
            total_analyzed_frames = 0
            no_face_frames = 0
            
            suspicious_movements = []
            detailed_analysis = []
            
            print(f"Starting optimized video analysis: {total_frames} frames at {fps:.1f} FPS, {duration:.2f} seconds")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break # End of video stream
                
                frame_count += 1
                
                # Apply frame analysis optimization: process only selected frames
                if frame_count % self.analysis_frame_skip != 0:
                    continue
                
                total_analyzed_frames += 1
                
                # Analyze the current frame
                frame_analysis = self.analyze_frame(frame)
                timestamp = frame_count / fps if fps > 0 else frame_count
                
                # Record detailed analysis (sample for efficiency, e.g., every 10th analyzed frame)
                frame_analysis['timestamp'] = timestamp
                frame_analysis['frame_number'] = frame_count
                
                if total_analyzed_frames % 10 == 0:
                    detailed_analysis.append(frame_analysis)
                
                # Count suspicious behavior
                if frame_analysis['looking_away']:
                    looking_away_frames += 1
                    
                    # Record significant movements for detailed reporting
                    if (frame_analysis['face_movement'] > self.face_movement_threshold or 
                        frame_analysis['eye_movement'] > self.eye_movement_threshold or
                        frame_analysis['faces_detected'] == 0):
                        
                        movement_data = {
                            'timestamp': timestamp,
                            'face_movement': frame_analysis['face_movement'],
                            'eye_movement': frame_analysis['eye_movement'],
                            'faces_detected': frame_analysis['faces_detected'],
                            'eyes_detected': frame_analysis['eyes_detected'],
                            'type': 'suspicious_behavior'
                        }
                        suspicious_movements.append(movement_data)
                
                if frame_analysis['faces_detected'] == 0:
                    no_face_frames += 1
                
                # Print progress (less frequent for performance)
                if total_analyzed_frames % 50 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Analysis progress: {progress:.1f}% ({total_analyzed_frames} frames analyzed)")
            
            cap.release() # Release video capture object
            
            # Calculate final metrics
            if total_analyzed_frames > 0:
                looking_away_percentage = (looking_away_frames / total_analyzed_frames) * 100
                no_face_percentage = (no_face_frames / total_analyzed_frames) * 100
                
                # Enhanced scoring logic for 24 FPS video
                base_cheating_score = looking_away_percentage * 0.7  # Base score from looking away
                no_face_penalty = no_face_percentage * 0.3  # Additional penalty for no face detection
                movement_penalty = min(25, len(suspicious_movements) * 2.5)  # Adjusted penalty for lower frame rate
                
                cheating_score = min(100, base_cheating_score + no_face_penalty + movement_penalty)
                
                # Determine if cheating is detected (adjusted threshold for 24 FPS)
                is_cheating = cheating_score > 30 # Threshold can be tuned
                
                # Compile analysis result dictionary
                analysis_result = {
                    'video_duration': duration,
                    'video_fps': fps,
                    'analysis_fps': fps / self.analysis_frame_skip,
                    'total_frames': total_frames,
                    'total_frames_analyzed': total_analyzed_frames,
                    'looking_away_frames': looking_away_frames,
                    'no_face_frames': no_face_frames,
                    'looking_away_percentage': looking_away_percentage,
                    'no_face_percentage': no_face_percentage,
                    'cheating_score': cheating_score,
                    'is_cheating_detected': is_cheating,
                    'suspicious_movements': suspicious_movements[:20],  # Store top 20 movements for report
                    'total_suspicious_movements': len(suspicious_movements),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'analysis_method': 'OpenCV Haar Cascades (24 FPS Optimized)',
                    'optimization_notes': f'Frame skip: {self.analysis_frame_skip}, Effective analysis rate: {fps/self.analysis_frame_skip:.1f} FPS'
                }
                
                # Save optimized detailed analysis (e.g., last 50 frame samples for debugging)
                detailed_analysis_file = os.path.join(output_dir, 'detailed_eye_analysis.json')
                with open(detailed_analysis_file, 'w') as f:
                    json.dump({
                        'summary': analysis_result,
                        'frame_samples': detailed_analysis[-50:] 
                    }, f, indent=2)
                
                # Save summary analysis results to eye_analysis.json
                analysis_file = os.path.join(output_dir, 'eye_analysis.json')
                with open(analysis_file, 'w') as f:
                    json.dump(analysis_result, f, indent=2)
                
                # Create human-readable report
                report_file = os.path.join(output_dir, 'cheating_analysis.txt')
                with open(report_file, 'w') as f:
                    f.write("OPTIMIZED OPENCV EYE TRACKING ANALYSIS REPORT\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Video Duration: {duration:.2f} seconds\n")
                    f.write(f"Video FPS: {fps:.2f}\n")
                    f.write(f"Analysis FPS: {fps/self.analysis_frame_skip:.2f} (optimized)\n")
                    f.write(f"Total Frames: {total_frames}\n")
                    f.write(f"Frames Analyzed: {total_analyzed_frames}\n")
                    f.write(f"Analysis Method: OpenCV Haar Cascades (24 FPS Optimized)\n\n")
                    
                    f.write("BEHAVIORAL ANALYSIS:\n")
                    f.write("-" * 25 + "\n")
                    f.write(f"Looking Away Percentage: {looking_away_percentage:.2f}%\n")
                    f.write(f"No Face Detection: {no_face_percentage:.2f}%\n")
                    f.write(f"Suspicious Movements: {len(suspicious_movements)}\n")
                    f.write(f"Cheating Score: {cheating_score:.2f}/100\n")
                    f.write(f"Cheating Detected: {'YES' if is_cheating else 'NO'}\n\n")
                    
                    f.write("OPTIMIZATION DETAILS:\n")
                    f.write("-" * 25 + "\n")
                    f.write(f"Frame Skip Rate: {self.analysis_frame_skip}\n")
                    f.write(f"Effective Analysis Rate: {fps/self.analysis_frame_skip:.1f} FPS\n")
                    f.write(f"Performance Gain: ~{self.analysis_frame_skip}x faster processing\n\n")
                    
                    if suspicious_movements:
                        f.write("TOP SUSPICIOUS MOMENTS:\n")
                        f.write("-" * 25 + "\n")
                        for i, movement in enumerate(suspicious_movements[:10]):
                            f.write(f"{i+1}. Time: {movement['timestamp']:.2f}s - ")
                            if movement['faces_detected'] == 0:
                                f.write("Face not detected\n")
                            else:
                                f.write(f"Movement detected (Face: {movement['face_movement']:.1f}px, Eyes: {movement['eye_movement']:.1f}px)\n")
                    
                    f.write(f"\nANALYSIS COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                print(f"Optimized analysis completed: Score {cheating_score:.2f}, Cheating: {is_cheating}")
                return analysis_result
            else:
                raise Exception("No frames could be analyzed from the video.")
                
        except Exception as e:
            print(f"Error in optimized eye tracking analysis: {e}")
            return {
                'error': str(e),
                'is_cheating_detected': False,
                'cheating_score': 0,
                'video_duration': 0,
                'total_frames_analyzed': 0,
                'analysis_timestamp': datetime.now().isoformat()
            }

