import tornado.ioloop
import tornado.web
import tornado.websocket
import numpy as np
import cv2
from deepface import DeepFace
import logging
import struct
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import asyncio
from typing import Optional, Dict, Any
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s'
)
logger = logging.getLogger(__name__)

class FaceAlignmentHandler(tornado.websocket.WebSocketHandler):
    executor = ThreadPoolExecutor(max_workers=4)
    
    def initialize(self) -> None:
        self.processing_lock = asyncio.Lock()
        
    def check_origin(self, origin: str) -> bool:
        logger.info(f"New connection request from origin: {origin}")
        return True

    def open(self) -> None:
        logger.info("New WebSocket connection opened")
        
    async def on_message(self, message: bytes) -> None:
        async with self.processing_lock:  # Ensure one processing per connection
            try:
                await self._process_message(message)
            except Exception as e:
                error_msg = f"Error processing image: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                await self.write_message(f"Error: {str(e)}")
                
    async def _process_message(self, message: bytes) -> None:
        # Extract message ID
        message_id = struct.unpack('!Q', message[:8])[0]
        logger.info(f"Processing message ID: {message_id}")
        
        # Process image in thread pool
        img = await self._decode_image(message[8:])
        face_objs = await self._detect_faces(img)
        
        if not face_objs:
            raise ValueError("No faces detected")
        
        # Prepare response
        response = await self._prepare_response(message_id, face_objs)
        
        # Send response
        await self.write_message(response, binary=True)
        logger.info(f"Response sent for message ID: {message_id}")
        
        # Clean up
        del img, face_objs
        gc.collect()

    async def _decode_image(self, image_data: bytes) -> np.ndarray:
        def _decode():
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            return img
            
        return await tornado.ioloop.IOLoop.current().run_in_executor(
            self.executor, _decode
        )

    async def _detect_faces(self, img: np.ndarray) -> list:
        def _detect():
            return DeepFace.extract_faces(
                img_path=img,
                detector_backend='retinaface',
                enforce_detection=False,
                align=True
            )
            
        return await tornado.ioloop.IOLoop.current().run_in_executor(
            self.executor, _detect
        )

    async def _prepare_response(self, message_id: int, face_objs: list) -> bytes:
        def _prepare():
            response_parts = [
                struct.pack('!Q', message_id),
                struct.pack('!I', len(face_objs))
            ]
            
            for idx, face_obj in enumerate(face_objs):
                aligned_face = self._normalize_face(face_obj['face'])
                face_bytes = self._encode_face(aligned_face)
                response_parts.extend([
                    struct.pack('!I', len(face_bytes)),
                    face_bytes
                ])
                
            return b''.join(response_parts)
            
        return await tornado.ioloop.IOLoop.current().run_in_executor(
            self.executor, _prepare
        )

    @staticmethod
    def _normalize_face(face: np.ndarray) -> np.ndarray:
        if face.dtype != np.uint8:
            face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX)
            face = face.astype(np.uint8)
        return cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _encode_face(face: np.ndarray) -> bytes:
        success, buffer = cv2.imencode('.png', face)
        if not success:
            raise ValueError("Failed to encode face")
        return buffer.tobytes()

    def on_close(self) -> None:
        logger.info("WebSocket connection closed")

def make_app() -> tornado.web.Application:
    return tornado.web.Application([
        (r"/align", FaceAlignmentHandler),
    ])

if __name__ == "__main__":
    try:
        app = make_app()
        port = 8888
        app.listen(port)
        logger.info(f"Face alignment server started on ws://localhost:{port}/align")
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        tornado.ioloop.IOLoop.current().stop()
    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
