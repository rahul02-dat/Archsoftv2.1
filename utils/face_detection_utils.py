import base64
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import os
from typing import Tuple, Optional, List
import pickle

class FaceRecognitionSystem:
    def __init__(self):
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Detection threshold - lower means stricter matching
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.45))
        self.liveness_threshold = float(os.getenv("LIVENESS_THRESHOLD", 0.7))
        
        print(f"✓ Automated Face Recognition System Initialized")
        print(f"  - Detection Threshold: {self.similarity_threshold}")
        print(f"  - Liveness Threshold: {self.liveness_threshold}")
        
    def base64_to_image(self, base64_string: str) -> np.ndarray:
        """Convert base64 string to OpenCV image"""
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image data")
        
        return img
    
    def detect_and_align(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect face and return normalized embedding"""
        faces = self.app.get(image)
        
        if len(faces) == 0:
            return None
        
        if len(faces) > 1:
            raise ValueError("Multiple faces detected. Please ensure only one face is visible.")
        
        face = faces[0]
        
        return face.normed_embedding
    
    def check_liveness(self, image: np.ndarray) -> Tuple[bool, float]:
        """Basic liveness detection using texture analysis"""
        faces = self.app.get(image)
        
        if len(faces) == 0:
            return False, 0.0
        
        face = faces[0]
        bbox = face.bbox
        x1, y1, x2, y2 = map(int, bbox)
        
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return False, 0.0
        
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        texture_score = min(laplacian_var / 100.0, 1.0)
        
        height, width = face_region.shape[:2]
        face_area = (x2 - x1) * (y2 - y1)
        image_area = image.shape[0] * image.shape[1]
        size_ratio = face_area / image_area
        
        size_score = 1.0 if 0.1 < size_ratio < 0.8 else 0.5
        
        liveness_score = (texture_score * 0.7 + size_score * 0.3)
        
        is_live = liveness_score >= self.liveness_threshold
        
        return is_live, liveness_score
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)
    
    def process_detection_image(
        self, 
        base64_image: str
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Process image for face detection
        
        Returns:
            (embedding, message)
        """
        try:
            image = self.base64_to_image(base64_image)
            
            # Liveness check
            is_live, liveness_score = self.check_liveness(image)
            if not is_live:
                return None, f"Liveness check failed. Score: {liveness_score:.2f}. Please ensure proper lighting and a real person."
            
            # Detect and extract embedding
            embedding = self.detect_and_align(image)
            if embedding is None:
                return None, "No face detected in the image. Please ensure face is clearly visible."
            
            return embedding, "Success"
            
        except ValueError as e:
            return None, str(e)
        except Exception as e:
            return None, f"Error processing image: {str(e)}"
    
    def verify_face(
        self, 
        query_embedding: np.ndarray, 
        stored_embeddings: List[bytes]
    ) -> Tuple[bool, float, Optional[int]]:
        """
        Verify face against stored embeddings
        
        Returns:
            (is_match, confidence_score, best_match_index)
        """
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
        """Serialize embedding to bytes for database storage"""
        return pickle.dumps(embedding)
    
    def deserialize_embedding(self, embedding_bytes: bytes) -> np.ndarray:
        """Deserialize embedding from bytes"""
        return pickle.loads(embedding_bytes)

# Global instance
face_recognition_system = FaceRecognitionSystem()