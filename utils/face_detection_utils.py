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
    
    def __init__(self, reference_image_path: str = "uniform_reference.jpg"):
        self.reference_hist = None
        self.uniform_threshold = float(os.getenv("UNIFORM_THRESHOLD", "0.4"))
        
        if os.path.exists(reference_image_path):
            self._load_reference_uniform(reference_image_path)
            print(f"✓ Uniform Reference Loaded: {reference_image_path}")
            print(f"  - Uniform Detection Threshold: {self.uniform_threshold}")
        else:
            print(f"⚠ Warning: Uniform reference image not found at {reference_image_path}")
            print(f"  - Employee detection will be disabled")
    
    def _load_reference_uniform(self, image_path: str):
        """Load reference uniform image and calculate its color histogram"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to HSV color space (better for color matching)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram for Hue and Saturation channels
            # Hue: 0-180, Saturation: 0-256
            hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            
            # Normalize the histogram
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            self.reference_hist = hist
            
        except Exception as e:
            print(f"✗ Error loading uniform reference: {str(e)}")
            self.reference_hist = None
    
    def _extract_body_region(self, image: np.ndarray, face_bbox: Dict) -> Optional[np.ndarray]:
        """Extract the torso/body region below the face for clothing analysis"""
        try:
            x1, y1, x2, y2 = face_bbox['x1'], face_bbox['y1'], face_bbox['x2'], face_bbox['y2']
            
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Estimate body region: below face, slightly wider, ~2x face height
            body_x1 = max(0, x1 - int(face_width * 0.2))
            body_x2 = min(image.shape[1], x2 + int(face_width * 0.2))
            body_y1 = y2  # Start right below the face
            body_y2 = min(image.shape[0], y2 + int(face_height * 2.0))
            
            # Extract body region
            body_region = image[body_y1:body_y2, body_x1:body_x2]
            
            if body_region.size == 0 or body_region.shape[0] < 20 or body_region.shape[1] < 20:
                return None
            
            return body_region
            
        except Exception as e:
            print(f"Error extracting body region: {str(e)}")
            return None
    
    def is_wearing_uniform(self, image: np.ndarray, face_bbox: Dict) -> Tuple[bool, float]:
        """
        Check if the person is wearing a uniform by comparing clothing color histogram
        
        Returns:
            Tuple[bool, float]: (is_employee, similarity_score)
        """
        if self.reference_hist is None:
            return False, 0.0
        
        try:
            # Extract body/torso region
            body_region = self._extract_body_region(image, face_bbox)
            
            if body_region is None:
                return False, 0.0
            
            # Convert to HSV
            hsv_body = cv2.cvtColor(body_region, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram
            body_hist = cv2.calcHist([hsv_body], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(body_hist, body_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # Compare histograms using correlation method
            # Returns value between -1 and 1, where 1 is perfect match
            similarity = cv2.compareHist(self.reference_hist, body_hist, cv2.HISTCMP_CORREL)
            
            # Ensure similarity is non-negative
            similarity = max(0.0, similarity)
            
            is_employee = similarity >= self.uniform_threshold
            
            return is_employee, float(similarity)
            
        except Exception as e:
            print(f"Error in uniform detection: {str(e)}")
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