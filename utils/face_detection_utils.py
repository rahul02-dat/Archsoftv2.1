import base64
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import os
from typing import Tuple, Optional, List, Dict
import pickle

class FaceRecognitionSystem:
    def __init__(self):
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.45))
        self.liveness_threshold = float(os.getenv("LIVENESS_THRESHOLD", 0.7))
        
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
            
            results.append({
                'embedding': face.normed_embedding,
                'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                'face_obj': face
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