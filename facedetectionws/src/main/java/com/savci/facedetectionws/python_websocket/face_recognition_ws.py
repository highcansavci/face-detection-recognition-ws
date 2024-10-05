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
import asyncio
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s'
)
logger = logging.getLogger(__name__)

class FaceRecognitionHandler(tornado.websocket.WebSocketHandler):
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
                # Send error as text message
                await self.write_message(error_msg)
                
    async def _process_message(self, message: bytes) -> None:
        # Extract message ID and image data
        message_id = struct.unpack('!Q', message[:8])[0]
        logger.info(f"Processing message ID: {message_id}")
        
        # Process image in thread pool
        img = await self._decode_image(message[8:])
        embedding = await self._get_embedding(img)
        
        if embedding is None:
            raise ValueError("Failed to generate embedding")
        
        # Prepare response with message ID and embedding
        response = await self._prepare_response(message_id, embedding)
        
        # Send response
        await self.write_message(response, binary=True)
        logger.info(f"Response sent for message ID: {message_id}")
        
        # Clean up
        del img, embedding
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

    async def _get_embedding(self, img: np.ndarray) -> np.ndarray:
        def _embed():
            embedding = DeepFace.represent(
                img_path=img,
                model_name="Facenet",
                detector_backend='retinaface',
                enforce_detection=False,
                align=True
            )
            return np.array(embedding[0]['embedding'], dtype=np.float32)
            
        return await tornado.ioloop.IOLoop.current().run_in_executor(
            self.executor, _embed
        )

    async def _prepare_response(self, message_id: int, embedding: np.ndarray) -> bytes:
        def _prepare():
            # Pack message ID and embedding into bytes
            message_id_bytes = struct.pack('!Q', message_id)
            embedding_bytes = embedding.tobytes()
            embedding_size = struct.pack('!I', len(embedding_bytes))
            
            return message_id_bytes + embedding_size + embedding_bytes
            
        return await tornado.ioloop.IOLoop.current().run_in_executor(
            self.executor, _prepare
        )

    def on_close(self) -> None:
        logger.info("WebSocket connection closed")

def make_app() -> tornado.web.Application:
    return tornado.web.Application([
        (r"/embed", FaceRecognitionHandler),
    ])

if __name__ == "__main__":
    try:
        app = make_app()
        port = 8889
        app.listen(port)
        logger.info(f"Face recognition server started on ws://localhost:{port}/embed")
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        tornado.ioloop.IOLoop.current().stop()
    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")