import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load the trained model
model = joblib.load('xgboost_model.pkl')  # Replace with your model file name

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def preprocess_hand_landmarks(hand_landmarks):
    """
    Extract all 21 hand landmarks (x, y, z) and return as a flattened list.
    """
    # Extract all landmarks
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    # Flatten the landmarks for model input
    return landmarks.flatten().reshape(1, -1)  # Return as a 2D array for model input

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Preprocess the hand landmarks for prediction
                features = preprocess_hand_landmarks(hand_landmarks)

                # Make predictions using the trained model
                prediction = model.predict(features)
                predicted_class = int(prediction[0])

                # Display the prediction on the frame
                cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Real-Time Hand Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()