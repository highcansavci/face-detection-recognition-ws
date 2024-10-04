from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import numpy as np

def detect_face(image_path):
    try:
        # Read the image
        img = cv2.imread(image_path)
        
        # Convert BGR to RGB (DeepFace works with RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect face
        face_objs = DeepFace.extract_faces(
                img_path=img,  # Input numpy array as image
                detector_backend='fastmtcnn',  # Use fast MTCNN for detection
                enforce_detection=False,
                align=True                            
            )
            
        if not face_objs:
            raise ValueError("No faces detected")

        # Get face region
        if len(face_objs) > 0:
            for idx, face in enumerate(face_objs):
                # Get the aligned face
                aligned_face = face['face']
                
                # Ensure aligned face is in uint8 format
                if aligned_face.dtype != np.uint8:
                    aligned_face = cv2.normalize(aligned_face, None, 0, 255, cv2.NORM_MINMAX)
                    aligned_face = aligned_face.astype(np.uint8)
                
                # Convert to BGR for saving
                if len(aligned_face.shape) == 3:  # If RGB
                    aligned_face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
                else:  # If grayscale
                    aligned_face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_GRAY2BGR)
                
                # Generate output filename
                output_filename = f"aligned_face_{idx+1}.png"
                
                # Save the aligned face
                cv2.imwrite(output_filename, aligned_face_bgr)

            print(f"Found {len(face_objs)} face(s) in the image")
        else:
            print("No faces detected in the image")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "/home/cansavci/Desktop/face-detection-ws/facedetectionws/src/main/java/com/savci/facedetectionws/client/can.png"
    detect_face(image_path)