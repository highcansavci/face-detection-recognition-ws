package com.savci.facedetectionws.client;

import org.springframework.web.socket.*;
import org.springframework.web.socket.client.WebSocketClient;
import org.springframework.web.socket.client.standard.StandardWebSocketClient;
import org.springframework.web.socket.handler.BinaryWebSocketHandler;

import jakarta.websocket.ContainerProvider;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;
import java.util.logging.Level;

import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.Mat;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;

public class FaceAlignmentClient implements Closeable {
    private static final Logger LOGGER = Logger.getLogger(FaceAlignmentClient.class.getName());
    private final WebSocketSession session;
    private final ConcurrentHashMap<Long, CompletableFuture<List<byte[]>>> responseMap = new ConcurrentHashMap<>();
    private final AtomicLong messageIdGenerator = new AtomicLong(0);

    public FaceAlignmentClient(String url) {
        try {
            LOGGER.info("Initializing WebSocket client for URL: " + url);
            WebSocketClient client = new StandardWebSocketClient();
            WebSocketHandler handler = new BinaryWebSocketHandler() {
                @Override
                public void afterConnectionEstablished(WebSocketSession session) {
                    LOGGER.info("WebSocket connection established with ID: " + session.getId());
                    session.setTextMessageSizeLimit(1048576);
                    session.setBinaryMessageSizeLimit(1048576);
                }

                @Override
                public void handleTransportError(WebSocketSession session, Throwable exception) {
                    LOGGER.severe("Transport error: " + exception.getMessage());
                    exception.printStackTrace();
                }

                @Override
                public void handleBinaryMessage(WebSocketSession session, BinaryMessage message) {
                    try {
                        byte[] payload = message.getPayload().array();
                        LOGGER.info("Received binary message of size: " + payload.length);

                        ByteBuffer buffer = ByteBuffer.wrap(payload);
                        long messageId = buffer.getLong();
                        LOGGER.info("Message ID from response: " + messageId);

                        int numFaces = buffer.getInt();
                        LOGGER.info("Number of faces in response: " + numFaces);

                        List<byte[]> faces = new ArrayList<>();
                        for (int i = 0; i < numFaces; i++) {
                            int faceSize = buffer.getInt();
                            LOGGER.info("Face " + (i + 1) + " size: " + faceSize);

                            byte[] faceData = new byte[faceSize];
                            buffer.get(faceData);
                            faces.add(faceData);
                            LOGGER.info("Face " + (i + 1) + " data read successfully");
                        }

                        CompletableFuture<List<byte[]>> future = responseMap.remove(messageId);
                        if (future != null) {
                            future.complete(faces);
                            LOGGER.info("Response processed successfully for message ID: " + messageId);
                        } else {
                            LOGGER.warning("No waiting future found for message ID: " + messageId);
                        }
                    } catch (Exception e) {
                        LOGGER.log(Level.SEVERE, "Error processing binary message", e);
                    }
                }

                @Override
                public void handleTextMessage(WebSocketSession session, TextMessage message) {
                    LOGGER.warning("Received text message (error): " + message.getPayload());
                    responseMap.values().forEach(
                            future -> future.completeExceptionally(new RuntimeException(message.getPayload())));
                    responseMap.clear();
                }
            };

            this.session = client.execute(handler, url).get(10, TimeUnit.SECONDS);
            LOGGER.info("WebSocket connection established successfully");
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Failed to connect to WebSocket server", e);
            throw new RuntimeException("Failed to connect to WebSocket server", e);
        }
    }

    public CompletableFuture<List<byte[]>> alignFaces(byte[] imageData) {
        if (!session.isOpen()) {
            throw new IllegalStateException("WebSocket session is not open");
        }

        CompletableFuture<List<byte[]>> responseFuture = new CompletableFuture<>();
        long messageId = messageIdGenerator.getAndIncrement();
        responseMap.put(messageId, responseFuture);

        try {
            LOGGER.info("Preparing to send image data for message ID: " + messageId);
            byte[] messageIdBytes = ByteBuffer.allocate(8).putLong(messageId).array();
            byte[] fullMessage = new byte[messageIdBytes.length + imageData.length];
            System.arraycopy(messageIdBytes, 0, fullMessage, 0, messageIdBytes.length);
            System.arraycopy(imageData, 0, fullMessage, messageIdBytes.length, imageData.length);

            LOGGER.info("Sending message of size: " + fullMessage.length);
            session.sendMessage(new BinaryMessage(fullMessage));
            LOGGER.info("Message sent successfully");
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Error sending message", e);
            responseMap.remove(messageId);
            responseFuture.completeExceptionally(e);
        }

        return responseFuture;
    }

    public Mat byteArrayToMat(byte[] imageData) {
        try {
            LOGGER.info("Converting byte array to Mat, size: " + imageData.length);
            Mat mat = opencv_imgcodecs.imdecode(new Mat(imageData), opencv_imgcodecs.IMREAD_COLOR);
            if (mat.empty()) {
                throw new RuntimeException("Failed to decode image data");
            }
            LOGGER.info("Successfully converted to Mat with size: " + mat.size().toString());
            return mat;
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error converting byte array to Mat", e);
            throw e;
        }
    }

    @Override
    public void close() throws IOException {
        if (session != null && session.isOpen()) {
            LOGGER.info("Closing WebSocket session");
            session.close();
        }
    }

    public static void main(String[] args) {
        try (FaceAlignmentClient client = new FaceAlignmentClient("ws://localhost:8888/align")) {
            LOGGER.info("Reading image file...");
            String imagePath = "facedetectionws/src/main/java/com/savci/facedetectionws/client/can.png";
            byte[] imageData = Files.readAllBytes(Paths.get(imagePath));
            LOGGER.info("Image file read successfully, size: " + imageData.length);

            LOGGER.info("Requesting face alignment...");
            List<byte[]> alignedFaces = client.alignFaces(imageData)
                    .get(30, TimeUnit.SECONDS); // Add timeout

            LOGGER.info("Received " + alignedFaces.size() + " aligned faces");

            for (int i = 0; i < alignedFaces.size(); i++) {
                String outputPath = String.format("aligned_face_%d.png", i + 1);
                Mat faceMat = client.byteArrayToMat(alignedFaces.get(i));
                imwrite(outputPath, faceMat);
                LOGGER.info("Saved aligned face: " + outputPath);
            }

            LOGGER.info("Processing completed successfully");
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error in main", e);
        }
    }
}