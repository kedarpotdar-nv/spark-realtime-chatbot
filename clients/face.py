"""Face recognition module using DeepFace."""

import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Lazy imports to avoid slow startup
_deepface = None
_cv2 = None


def _get_deepface():
    global _deepface
    if _deepface is None:
        from deepface import DeepFace
        _deepface = DeepFace
    return _deepface


def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


class FaceRecognizer:
    """Face detection and recognition using DeepFace."""

    def __init__(self, db_path: str = "face_db.json", model_name: str = "Facenet512"):
        """Initialize face recognizer.

        Args:
            db_path: Path to JSON file storing face embeddings
            model_name: DeepFace model to use (Facenet512, VGG-Face, ArcFace, etc.)
        """
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.faces_db: Dict[str, dict] = {}  # name -> {embedding, enrolled_at}
        self._load_db()
        print(f"[Face] Initialized with {len(self.faces_db)} enrolled faces")

    def _load_db(self):
        """Load face database from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    self.faces_db = json.load(f)
                print(f"[Face] Loaded {len(self.faces_db)} faces from {self.db_path}")
            except Exception as e:
                print(f"[Face] Error loading DB: {e}")
                self.faces_db = {}

    def _save_db(self):
        """Save face database to disk."""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.faces_db, f, indent=2)
            print(f"[Face] Saved {len(self.faces_db)} faces to {self.db_path}")
        except Exception as e:
            print(f"[Face] Error saving DB: {e}")

    def _base64_to_image(self, image_base64: str) -> np.ndarray:
        """Convert base64 image to numpy array."""
        cv2 = _get_cv2()

        # Handle data URL format
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]

        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image

    def detect_faces(self, image_base64: str) -> List[dict]:
        """Detect faces in an image.

        Args:
            image_base64: Base64 encoded image

        Returns:
            List of detected faces with bounding boxes
        """
        DeepFace = _get_deepface()

        try:
            image = self._base64_to_image(image_base64)

            # Detect faces
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend='opencv',  # Fast detector
                enforce_detection=False
            )

            results = []
            for face in faces:
                if face.get('confidence', 0) > 0.5:
                    results.append({
                        'bbox': face.get('facial_area', {}),
                        'confidence': face.get('confidence', 0)
                    })

            return results

        except Exception as e:
            print(f"[Face] Detection error: {e}")
            return []

    def get_embedding(self, image_base64: str) -> Optional[List[float]]:
        """Extract face embedding from image.

        Args:
            image_base64: Base64 encoded image (should contain one face)

        Returns:
            Face embedding vector or None if no face found
        """
        DeepFace = _get_deepface()

        try:
            image = self._base64_to_image(image_base64)

            # Get embedding
            embeddings = DeepFace.represent(
                img_path=image,
                model_name=self.model_name,
                detector_backend='opencv',
                enforce_detection=False
            )

            if embeddings and len(embeddings) > 0:
                return embeddings[0]['embedding']

            return None

        except Exception as e:
            print(f"[Face] Embedding error: {e}")
            return None

    def enroll_face(self, name: str, image_base64: str) -> bool:
        """Enroll a new face.

        Args:
            name: Name to associate with the face
            image_base64: Base64 encoded image

        Returns:
            True if enrollment successful
        """
        embedding = self.get_embedding(image_base64)

        if embedding is None:
            print(f"[Face] No face found for enrollment: {name}")
            return False

        from datetime import datetime
        self.faces_db[name] = {
            'embedding': embedding,
            'enrolled_at': datetime.now().isoformat()
        }
        self._save_db()
        print(f"[Face] Enrolled: {name}")
        return True

    def recognize_faces(self, image_base64: str, threshold: float = 0.6) -> List[dict]:
        """Recognize faces in an image.

        Args:
            image_base64: Base64 encoded image
            threshold: Distance threshold for recognition (lower = stricter)

        Returns:
            List of recognized faces with names and confidence
        """
        DeepFace = _get_deepface()

        if not self.faces_db:
            return []

        try:
            image = self._base64_to_image(image_base64)

            # Get all face embeddings in the image
            embeddings = DeepFace.represent(
                img_path=image,
                model_name=self.model_name,
                detector_backend='opencv',
                enforce_detection=False
            )

            results = []
            for face_data in embeddings:
                embedding = face_data.get('embedding')
                facial_area = face_data.get('facial_area', {})

                if embedding is None:
                    continue

                # Compare against enrolled faces
                best_match = None
                best_distance = float('inf')

                for name, data in self.faces_db.items():
                    stored_embedding = data['embedding']
                    distance = self._cosine_distance(embedding, stored_embedding)

                    if distance < best_distance:
                        best_distance = distance
                        best_match = name

                if best_match and best_distance < threshold:
                    results.append({
                        'name': best_match,
                        'confidence': 1 - best_distance,
                        'bbox': facial_area
                    })
                else:
                    results.append({
                        'name': 'Unknown',
                        'confidence': 0,
                        'bbox': facial_area
                    })

            return results

        except Exception as e:
            print(f"[Face] Recognition error: {e}")
            return []

    def _cosine_distance(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine distance between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def delete_face(self, name: str) -> bool:
        """Remove an enrolled face.

        Args:
            name: Name of the face to remove

        Returns:
            True if face was removed
        """
        if name in self.faces_db:
            del self.faces_db[name]
            self._save_db()
            print(f"[Face] Deleted: {name}")
            return True
        return False

    def list_enrolled(self) -> List[str]:
        """Get list of enrolled face names."""
        return list(self.faces_db.keys())

    def format_scene_description(self, recognized_faces: List[dict]) -> str:
        """Format recognized faces into a scene description for VLM context.

        Args:
            recognized_faces: Output from recognize_faces()

        Returns:
            Human-readable scene description
        """
        if not recognized_faces:
            return ""

        known = [f for f in recognized_faces if f['name'] != 'Unknown']
        unknown_count = len([f for f in recognized_faces if f['name'] == 'Unknown'])

        parts = []
        if known:
            names = [f"{f['name']} ({f['confidence']*100:.0f}%)" for f in known]
            parts.append(f"Recognized: {', '.join(names)}")

        if unknown_count > 0:
            parts.append(f"{unknown_count} unknown person(s)")

        return "[People in frame: " + "; ".join(parts) + "]"


# Global instance (lazy loaded)
_face_recognizer: Optional[FaceRecognizer] = None


def get_face_recognizer() -> FaceRecognizer:
    """Get or create the global face recognizer instance."""
    global _face_recognizer
    if _face_recognizer is None:
        _face_recognizer = FaceRecognizer()
    return _face_recognizer
