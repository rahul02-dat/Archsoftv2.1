import base64
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import os
from typing import Tuple, Optional, List, Dict
import pickle

class UniformMatcher:
    """Analyzes clothing color to detect employees wearing uniforms"""
    
    def __init__(self, reference_image_path: str = "uniform_refrence.jpg"):
        self.reference_hist = None
        self.reference_color_ranges = None  # Store dominant color ranges
        self.uniform_threshold = float(os.getenv("UNIFORM_THRESHOLD", "0.35"))  # Lowered from 0.4
        self.enabled = False
        
        # Check absolute path
        abs_path = os.path.abspath(reference_image_path)
        print(f"\n{'='*60}")
        print(f"UNIFORM DETECTION INITIALIZATION")
        print(f"{'='*60}")
        print(f"Looking for: {abs_path}")
        
        if os.path.exists(reference_image_path):
            if self._load_reference_uniform(reference_image_path):
                self.enabled = True
                print(f"✓ Uniform Detection ENABLED")
                print(f"  - Reference: {reference_image_path}")
                print(f"  - Threshold: {self.uniform_threshold}")
            else:
                print(f"✗ Uniform Detection FAILED to initialize")
        else:
            print(f"✗ Reference image NOT FOUND")
            print(f"  - Employee detection DISABLED")
            print(f"\n  To enable uniform detection:")
            print(f"  1. Place uniform image at: {abs_path}")
            print(f"  2. Restart the application")
        print(f"{'='*60}\n")
    
    def _load_reference_uniform(self, image_path: str):
        """Load reference uniform image and calculate its color histogram"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"✗ Error: Could not read image file")
                return False
            
            print(f"✓ Image loaded: {img.shape[1]}x{img.shape[0]} pixels")
            
            # Convert to HSV color space (better for color matching)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Calculate multiple histograms for better matching
            # 1. Full HSV histogram (Hue + Saturation)
            hist_hs = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist_hs, hist_hs, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # 2. Individual channel histograms
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_v, hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # Store all histograms
            self.reference_hist = {
                'hs': hist_hs,
                'h': hist_h,
                's': hist_s,
                'v': hist_v
            }
            
            # Extract dominant color information
            avg_hue = np.mean(hsv[:,:,0])
            avg_sat = np.mean(hsv[:,:,1])
            avg_val = np.mean(hsv[:,:,2])
            
            std_hue = np.std(hsv[:,:,0])
            std_sat = np.std(hsv[:,:,1])
            
            # Store color ranges for additional matching
            self.reference_color_ranges = {
                'hue_mean': avg_hue,
                'hue_std': std_hue,
                'sat_mean': avg_sat,
                'sat_std': std_sat,
                'val_mean': avg_val
            }
            
            print(f"✓ Reference histogram calculated")
            print(f"  - Avg Hue: {avg_hue:.1f}°, Std: {std_hue:.1f}")
            print(f"  - Avg Sat: {avg_sat:.1f}, Std: {std_sat:.1f}")
            print(f"  - Avg Val: {avg_val:.1f}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading uniform reference: {str(e)}")
            self.reference_hist = None
            self.reference_color_ranges = None
            return False
    
    def _extract_body_region(self, image: np.ndarray, face_bbox: Dict) -> Optional[np.ndarray]:
        """
        Extract the torso/body region below the face for clothing analysis.
        Uses a focused approach to capture only the central chest area.
        """
        try:
            x1, y1, x2, y2 = face_bbox['x1'], face_bbox['y1'], face_bbox['x2'], face_bbox['y2']
            
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Strategy: Sample multiple regions and combine them
            # Region 1: Narrow center torso (most reliable)
            center_x1 = max(0, x1 + int(face_width * 0.25))
            center_x2 = min(image.shape[1], x2 - int(face_width * 0.25))
            center_y1 = y2 + int(face_height * 0.2)  # Small gap below face
            center_y2 = min(image.shape[0], y2 + int(face_height * 1.5))
            
            # Validate bounds
            if center_x2 <= center_x1 or center_y2 <= center_y1:
                print("Warning: Body region dimensions invalid")
                return None
            
            # Extract center body region
            body_region = image[center_y1:center_y2, center_x1:center_x2]
            
            if body_region.size == 0 or body_region.shape[0] < 15 or body_region.shape[1] < 15:
                return None
            
            # Debug: Save extracted region (optional, comment out in production)
            # cv2.imwrite('debug_body_region.jpg', body_region)
            
            return body_region
            
        except Exception as e:
            print(f"Error extracting body region: {str(e)}")
            return None
    
    def _compare_color_statistics(self, body_hsv: np.ndarray) -> float:
        """
        Compare color statistics (mean/std) between reference and detected body
        Returns a similarity score between 0 and 1
        """
        if self.reference_color_ranges is None:
            return 0.0
        
        try:
            # Calculate stats for detected body
            body_hue_mean = np.mean(body_hsv[:,:,0])
            body_sat_mean = np.mean(body_hsv[:,:,1])
            body_val_mean = np.mean(body_hsv[:,:,2])
            
            # Calculate differences (normalized)
            hue_diff = abs(body_hue_mean - self.reference_color_ranges['hue_mean']) / 180.0
            sat_diff = abs(body_sat_mean - self.reference_color_ranges['sat_mean']) / 255.0
            val_diff = abs(body_val_mean - self.reference_color_ranges['val_mean']) / 255.0
            
            # For hue, handle circular nature (e.g., 179 and 1 are close)
            if hue_diff > 0.5:
                hue_diff = 1.0 - hue_diff
            
            # Similarity score (inverse of difference)
            hue_similarity = 1.0 - hue_diff
            sat_similarity = 1.0 - sat_diff
            val_similarity = 1.0 - val_diff
            
            # Weighted combination: Hue is most important, then Saturation
            color_similarity = (hue_similarity * 0.5 + sat_similarity * 0.3 + val_similarity * 0.2)
            
            return color_similarity
            
        except Exception as e:
            print(f"Error in color statistics comparison: {str(e)}")
            return 0.0
    
    def is_wearing_uniform(self, image: np.ndarray, face_bbox: Dict) -> Tuple[bool, float]:
        """
        Check if the person is wearing a uniform by comparing clothing color
        Uses multiple methods: histogram comparison + color statistics
        
        Returns:
            Tuple[bool, float]: (is_employee, similarity_score)
        """
        if self.reference_hist is None or not self.enabled:
            return False, 0.0
        
        try:
            # Extract body/torso region
            body_region = self._extract_body_region(image, face_bbox)
            
            if body_region is None:
                return False, 0.0
            
            # Convert to HSV
            hsv_body = cv2.cvtColor(body_region, cv2.COLOR_BGR2HSV)
            
            # Method 1: Histogram comparison (H+S combined)
            body_hist_hs = cv2.calcHist([hsv_body], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(body_hist_hs, body_hist_hs, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            hist_similarity = cv2.compareHist(self.reference_hist['hs'], body_hist_hs, cv2.HISTCMP_CORREL)
            hist_similarity = max(0.0, hist_similarity)
            
            # Method 2: Individual channel comparisons
            body_hist_h = cv2.calcHist([hsv_body], [0], None, [180], [0, 180])
            body_hist_s = cv2.calcHist([hsv_body], [1], None, [256], [0, 256])
            
            cv2.normalize(body_hist_h, body_hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(body_hist_s, body_hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            h_similarity = max(0.0, cv2.compareHist(self.reference_hist['h'], body_hist_h, cv2.HISTCMP_CORREL))
            s_similarity = max(0.0, cv2.compareHist(self.reference_hist['s'], body_hist_s, cv2.HISTCMP_CORREL))
            
            # Method 3: Color statistics comparison
            color_stat_similarity = self._compare_color_statistics(hsv_body)
            
            # Combine all methods with weights
            # Histogram methods are strong, color stats provide additional validation
            combined_similarity = (
                hist_similarity * 0.4 +
                h_similarity * 0.25 +
                s_similarity * 0.15 +
                color_stat_similarity * 0.2
            )
            
            is_employee = combined_similarity >= self.uniform_threshold
            
            # Enhanced debug logging
            print(f"\n  ═══ Uniform Detection Analysis ═══")
            print(f"  Combined Histogram: {hist_similarity:.3f}")
            print(f"  Hue Channel: {h_similarity:.3f}")
            print(f"  Saturation Channel: {s_similarity:.3f}")
            print(f"  Color Statistics: {color_stat_similarity:.3f}")
            print(f"  ─────────────────────────────────")
            print(f"  FINAL SCORE: {combined_similarity:.3f}")
            print(f"  Threshold: {self.uniform_threshold:.3f}")
            print(f"  Result: {'✓ EMPLOYEE' if is_employee else '✗ Customer'}")
            print(f"  ═══════════════════════════════════\n")
            
            return is_employee, float(combined_similarity)
            
        except Exception as e:
            print(f"Error in uniform detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, 0.0


class FaceRecognitionSystem:
    def __init__(self):
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.45))
        self.liveness_threshold = float(os.getenv("LIVENESS_THRESHOLD", 0.7))
        
        # Initialize uniform matcher
        self.uniform_matcher = UniformMatcher()
        
        print(f"✓ Automated Face Recognition System Initialized")
        print(f"  - Detection Threshold: {self.similarity_threshold}")
        print(f"  - Liveness Threshold: {self.liveness_threshold}")
        
    def base64_to_image(self, base64_string: str) -> np.ndarray:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image data")
        
        return img
    
    def detect_all_faces(self, image: np.ndarray) -> List[Dict]:
        faces = self.app.get(image)
        
        if len(faces) == 0:
            return []
        
        results = []
        for face in faces:
            bbox = face.bbox
            x1, y1, x2, y2 = map(int, bbox)
            
            bbox_dict = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            
            # Check if person is wearing uniform
            is_employee, uniform_score = self.uniform_matcher.is_wearing_uniform(image, bbox_dict)
            
            results.append({
                'embedding': face.normed_embedding,
                'bbox': bbox_dict,
                'face_obj': face,
                'is_employee': is_employee,
                'uniform_score': uniform_score
            })
        
        return results
    
    def check_face_liveness(self, image: np.ndarray, bbox: Dict) -> Tuple[bool, float]:
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return False, 0.0
        
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        texture_score = min(laplacian_var / 100.0, 1.0)
        
        face_area = (x2 - x1) * (y2 - y1)
        image_area = image.shape[0] * image.shape[1]
        size_ratio = face_area / image_area
        
        size_score = 1.0 if 0.1 < size_ratio < 0.8 else 0.5
        
        liveness_score = (texture_score * 0.7 + size_score * 0.3)
        
        is_live = liveness_score >= self.liveness_threshold
        
        return is_live, liveness_score
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)
    
    def process_detection_image(
        self, 
        base64_image: str
    ) -> Tuple[Optional[List[Dict]], str]:
        try:
            image = self.base64_to_image(base64_image)
            
            face_results = self.detect_all_faces(image)
            
            if len(face_results) == 0:
                return None, "No faces detected in the image."
            
            valid_faces = []
            for face_data in face_results:
                is_live, liveness_score = self.check_face_liveness(image, face_data['bbox'])
                
                if is_live:
                    valid_faces.append(face_data)
            
            if len(valid_faces) == 0:
                return None, "Liveness check failed for all detected faces."
            
            return valid_faces, "Success"
            
        except ValueError as e:
            return None, str(e)
        except Exception as e:
            return None, f"Error processing image: {str(e)}"
    
    def verify_face(
        self, 
        query_embedding: np.ndarray, 
        stored_embeddings: List[bytes]
    ) -> Tuple[bool, float, Optional[int]]:
        max_similarity = -1.0
        best_match_idx = None
        
        for idx, stored_embedding_bytes in enumerate(stored_embeddings):
            stored_embedding = pickle.loads(stored_embedding_bytes)
            
            similarity = self.compute_similarity(query_embedding, stored_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_idx = idx
        
        is_match = max_similarity >= self.similarity_threshold
        
        return is_match, max_similarity, best_match_idx
    
    def serialize_embedding(self, embedding: np.ndarray) -> bytes:
        return pickle.dumps(embedding)
    
    def deserialize_embedding(self, embedding_bytes: bytes) -> np.ndarray:
        return pickle.loads(embedding_bytes)

face_recognition_system = FaceRecognitionSystem()