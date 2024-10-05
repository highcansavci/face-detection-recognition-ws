package com.savci.facedetectionws.client;

import org.springframework.web.socket.*;
import org.springframework.web.socket.client.WebSocketClient;
import org.springframework.web.socket.client.standard.StandardWebSocketClient;
import org.springframework.web.socket.handler.BinaryWebSocketHandler;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;
import java.util.logging.Level;

public class FaceProcessingClient implements Closeable {
    private static final Logger LOGGER = Logger.getLogger(FaceProcessingClient.class.getName());
    private final WebSocketSession alignmentSession;
    private final WebSocketSession recognitionSession;
    private final ConcurrentHashMap<Long, CompletableFuture<byte[]>> alignmentResponseMap = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Long, CompletableFuture<float[]>> recognitionResponseMap = new ConcurrentHashMap<>();
    private final AtomicLong messageIdGenerator = new AtomicLong(0);

    public FaceProcessingClient(String alignmentUrl, String recognitionUrl) {
        try {
            WebSocketClient client = new StandardWebSocketClient();

            // Initialize alignment session
            this.alignmentSession = client.execute(createAlignmentHandler(), alignmentUrl)
                    .get(10, TimeUnit.SECONDS);
            LOGGER.info("Alignment WebSocket connection established");

            // Initialize recognition session
            this.recognitionSession = client.execute(createRecognitionHandler(), recognitionUrl)
                    .get(10, TimeUnit.SECONDS);
            LOGGER.info("Recognition WebSocket connection established");
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Failed to connect to WebSocket servers", e);
            throw new RuntimeException("Failed to connect to WebSocket servers", e);
        }
    }

    private WebSocketHandler createAlignmentHandler() {
        return new BinaryWebSocketHandler() {
            @Override
            public void afterConnectionEstablished(WebSocketSession session) {
                LOGGER.info("WebSocket connection established with ID: " + session.getId());
                session.setTextMessageSizeLimit(1048576);
                session.setBinaryMessageSizeLimit(1048576);
            }

            @Override
            public void handleBinaryMessage(WebSocketSession session, BinaryMessage message) {
                try {
                    ByteBuffer buffer = message.getPayload();
                    buffer.order(ByteOrder.BIG_ENDIAN);

                    long messageId = buffer.getLong();
                    int numFaces = buffer.getInt();

                    if (numFaces != 1) {
                        throw new RuntimeException("Expected exactly one face, found " + numFaces);
                    }

                    int faceSize = buffer.getInt();
                    byte[] faceData = new byte[faceSize];
                    buffer.get(faceData);

                    CompletableFuture<byte[]> future = alignmentResponseMap.remove(messageId);
                    if (future != null) {
                        future.complete(faceData);
                        LOGGER.info("Alignment processed for message ID: " + messageId);
                    }
                } catch (Exception e) {
                    LOGGER.log(Level.SEVERE, "Error processing alignment message", e);
                }
            }

            @Override
            public void handleTextMessage(WebSocketSession session, TextMessage message) {
                LOGGER.warning("Alignment error: " + message.getPayload());
                alignmentResponseMap.values()
                        .forEach(future -> future.completeExceptionally(new RuntimeException(message.getPayload())));
                alignmentResponseMap.clear();
            }
        };
    }

    private WebSocketHandler createRecognitionHandler() {
        return new BinaryWebSocketHandler() {
            @Override
            public void afterConnectionEstablished(WebSocketSession session) {
                LOGGER.info("WebSocket connection established with ID: " + session.getId());
                session.setTextMessageSizeLimit(1048576);
                session.setBinaryMessageSizeLimit(1048576);
            }

            @Override
            public void handleBinaryMessage(WebSocketSession session, BinaryMessage message) {
                try {
                    ByteBuffer buffer = message.getPayload();
                    buffer.order(ByteOrder.BIG_ENDIAN);

                    long messageId = buffer.getLong();
                    int embeddingSize = buffer.getInt();

                    float[] embedding = new float[embeddingSize / 4];
                    buffer.asFloatBuffer().get(embedding);

                    CompletableFuture<float[]> future = recognitionResponseMap.remove(messageId);
                    if (future != null) {
                        future.complete(embedding);
                        LOGGER.info("Recognition processed for message ID: " + messageId);
                    }
                } catch (Exception e) {
                    LOGGER.log(Level.SEVERE, "Error processing recognition message", e);
                }
            }

            @Override
            public void handleTextMessage(WebSocketSession session, TextMessage message) {
                LOGGER.warning("Recognition error: " + message.getPayload());
                recognitionResponseMap.values()
                        .forEach(future -> future.completeExceptionally(new RuntimeException(message.getPayload())));
                recognitionResponseMap.clear();
            }
        };
    }

    public CompletableFuture<float[]> processImage(byte[] imageData) {
        return alignFace(imageData)
                .thenCompose(this::getEmbedding);
    }

    private CompletableFuture<byte[]> alignFace(byte[] imageData) {
        if (!alignmentSession.isOpen()) {
            throw new IllegalStateException("Alignment WebSocket session is not open");
        }

        CompletableFuture<byte[]> future = new CompletableFuture<>();
        long messageId = messageIdGenerator.getAndIncrement();
        alignmentResponseMap.put(messageId, future);

        try {
            byte[] messageIdBytes = ByteBuffer.allocate(8)
                    .order(ByteOrder.BIG_ENDIAN)
                    .putLong(messageId)
                    .array();

            byte[] fullMessage = new byte[messageIdBytes.length + imageData.length];
            System.arraycopy(messageIdBytes, 0, fullMessage, 0, messageIdBytes.length);
            System.arraycopy(imageData, 0, fullMessage, messageIdBytes.length, imageData.length);

            alignmentSession.sendMessage(new BinaryMessage(fullMessage));
            LOGGER.info("Alignment request sent for message ID: " + messageId);
        } catch (IOException e) {
            alignmentResponseMap.remove(messageId);
            future.completeExceptionally(e);
        }

        return future;
    }

    private CompletableFuture<float[]> getEmbedding(byte[] alignedFaceData) {
        if (!recognitionSession.isOpen()) {
            throw new IllegalStateException("Recognition WebSocket session is not open");
        }

        CompletableFuture<float[]> future = new CompletableFuture<>();
        long messageId = messageIdGenerator.getAndIncrement();
        recognitionResponseMap.put(messageId, future);

        try {
            byte[] messageIdBytes = ByteBuffer.allocate(8)
                    .order(ByteOrder.BIG_ENDIAN)
                    .putLong(messageId)
                    .array();

            byte[] fullMessage = new byte[messageIdBytes.length + alignedFaceData.length];
            System.arraycopy(messageIdBytes, 0, fullMessage, 0, messageIdBytes.length);
            System.arraycopy(alignedFaceData, 0, fullMessage, messageIdBytes.length, alignedFaceData.length);

            recognitionSession.sendMessage(new BinaryMessage(fullMessage));
            LOGGER.info("Recognition request sent for message ID: " + messageId);
        } catch (IOException e) {
            recognitionResponseMap.remove(messageId);
            future.completeExceptionally(e);
        }

        return future;
    }

    @Override
    public void close() throws IOException {
        if (alignmentSession != null && alignmentSession.isOpen()) {
            alignmentSession.close();
        }
        if (recognitionSession != null && recognitionSession.isOpen()) {
            recognitionSession.close();
        }
    }

    public static void main(String[] args) {
        try (FaceProcessingClient client = new FaceProcessingClient(
                "ws://localhost:8888/align",
                "ws://localhost:8889/embed")) {

            String imagePath = "facedetectionws/src/main/java/com/savci/facedetectionws/client/can.png";
            byte[] imageData = Files.readAllBytes(Paths.get(imagePath));

            float[] embedding = client.processImage(imageData)
                    .get(30, TimeUnit.SECONDS);

            LOGGER.info("Received embedding of size: " + embedding.length);

            // Use the embedding as needed
            for (int i = 0; i < Math.min(5, embedding.length); i++) {
                LOGGER.info("Embedding[" + i + "]: " + embedding[i]);
            }

        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error in main", e);
        }
    }
}