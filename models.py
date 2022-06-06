import cv2
import mediapipe as mp

class Hands:
    def __init__(self, max_num_hands=10):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=max_num_hands, min_detection_confidence=0.5)

    def detect(self, img):
        # Convert the BGR image to RGB before processing.
        results = self.hands.process(img)

        # Print handedness and draw hand landmarks on the image.
        if not results.multi_hand_landmarks:
            return img
        image_height, image_width, _ = img.shape
        annotated_image = img.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style())
        return annotated_image

class BBoxFace:
    def __init__(self, max_num_hands=10):
        self.mp_face = mp.solutions.face_detection
        self.face = self.mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    def detect(self, img):
        # Convert the BGR image to RGB before processing.
        results = self.face.process(img)
	
        # Print handedness and draw hand landmarks on the image.
        if not results.detections:
            return img
        image_height, image_width, _ = img.shape
        annotated_image = img.copy()
        for detection in results.detections:
            coords = detection.location_data.relative_bounding_box
            x1 = coords.xmin*image_width
            y1 = coords.ymin*image_height
            x2 = x1+coords.width*image_width
            y2 = y1+coords.height*image_height
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)

        return annotated_image

class FaceDetection:
    def __init__(self, max_num_faces=10):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=max_num_faces,min_detection_confidence=0.5)
	  
    def detect(self, img):
        # Convert the BGR image to RGB before processing.
        results = self.face_mesh.process(img)

        # Print handedness and draw hand landmarks on the image.
        annotated_image = img.copy()
        if not results.multi_face_landmarks:
            return annotated_image
        for face_landmarks in results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
            self.mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
        return annotated_image

