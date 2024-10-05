import tornado.ioloop
import tornado.web
import tornado.websocket
import numpy as np
import cv2
from deepface import DeepFace
import logging
import struct
import traceback

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaceAlignmentHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        logger.info(f"New connection request from origin: {origin}")
        return True

    def open(self):
        logger.info("New WebSocket connection opened")
        
    async def on_message(self, message):
        try:
            logger.info(f"Received message of length: {len(message)} bytes")
            
            # Extract message ID (first 8 bytes)
            message_id = struct.unpack('!Q', message[:8])[0]
            logger.info(f"Processing message ID: {message_id}")
            
            # Convert remaining bytes to numpy array
            nparr = np.frombuffer(message[8:], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image")
                
            logger.info(f"Successfully decoded image with shape: {img.shape}")

            # Perform face detection and alignment
            logger.info("Starting face detection and alignment...")
            face_objs = DeepFace.extract_faces(
                img_path=img,
                detector_backend='retinaface',
                enforce_detection=False,
                align=True
            )

            if not face_objs:
                raise ValueError("No faces detected")

            logger.info(f"Detected {len(face_objs)} faces")

            # Prepare response buffer
            response_parts = []
            response_parts.append(struct.pack('!Q', message_id))
            response_parts.append(struct.pack('!I', len(face_objs)))
            
            # Process each face
            for idx, face_obj in enumerate(face_objs):
                aligned_face = face_obj['face']
                logger.debug(f"Processing face {idx + 1}, shape: {aligned_face.shape}")

                # Ensure aligned face is in uint8 format
                if aligned_face.dtype != np.uint8:
                    aligned_face = cv2.normalize(aligned_face, None, 0, 255, cv2.NORM_MINMAX)
                    aligned_face = aligned_face.astype(np.uint8)

                aligned_face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
                
                is_success, buffer = cv2.imencode(".png", aligned_face_bgr)
                if not is_success:
                    raise ValueError(f"Failed to encode face {idx + 1}")
                
                face_bytes = buffer.tobytes()
                response_parts.append(struct.pack('!I', len(face_bytes)))
                response_parts.append(face_bytes)
                logger.debug(f"Face {idx + 1} encoded, size: {len(face_bytes)} bytes")

            # Combine all parts into final response
            response = b''.join(response_parts)
            logger.info(f"Sending response of {len(response)} bytes")
            
            # Send response
            await self.write_message(response, binary=True)
            logger.info(f"Response sent successfully for message ID: {message_id}")

        except Exception as e:
            error_msg = f"Error processing image: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            await self.write_message(f"Error: {str(e)}")

def make_app():
    return tornado.web.Application([
        (r"/align", FaceAlignmentHandler),
    ])

if __name__ == "__main__":
    try:
        app = make_app()
        port = 8888
        app.listen(port)
        logger.info(f"Face alignment WebSocket server started on ws://localhost:{port}/align")
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        tornado.ioloop.IOLoop.current().stop()
    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")