import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define the header for the CSV file
header = ['number']
for i in range(21):  # MediaPipe provides 21 landmarks
    header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])

# Create or open the CSV file
filename = 'datasetall.csv'
if not os.path.exists(filename):
    with open(filename, 'w', newline="") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(header)  # Write the header if the file doesn't exist

# Process images in folders 0 to 9
count = 0
for folder in range(10):  # Loop through folders named 0 to 9
    folder_path = str(folder)  # Folder name as a string
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist. Skipping...")
        continue

    for images in os.listdir(folder_path):  # Iterate through images in the folder
        print(f"Processing {images} in folder {folder_path}...")
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:
            
            # Read and process the image
            image_path = os.path.join(folder_path, images)
            image = cv2.flip(cv2.imread(image_path), 1)  # Flip for correct handedness
            if image is None:
                print(f"Error reading image: {image_path}")
                continue
            
            # Convert the BGR image to RGB
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.multi_hand_landmarks:
                continue

            for hand_landmarks in results.multi_hand_landmarks:
                # Extract all 21 landmarks
                row = [folder]  # Use the folder name (0-9) as the label
                for i, landmark in enumerate(hand_landmarks.landmark):
                    row.extend([landmark.x, landmark.y, landmark.z])

                # Write the data to the CSV file
                with open(filename, 'a', newline='') as f_object:
                    writer_object = csv.writer(f_object)
                    writer_object.writerow(row)

                # Draw the hand landmarks on the image
                annotated_image = image.copy()
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Save the annotated image (optional)
                output_folder = "annotated_images"
                os.makedirs(output_folder, exist_ok=True)
                cv2.imwrite(os.path.join(output_folder, f"{folder}_{images}"), annotated_image)
                count += 1

print(f"Total images processed: {count}")