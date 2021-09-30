import cv2
import imutils
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


def detect_mask(frame, face_net, mask_net):
    """Takes the frame, the face detector model, and the classifier model.
        Returns coordinates for locations of faces and probabilities of each class for the face"""

    # Grab dimensions of frame
    (h, w) = frame.shape[:2]

    # Construct a blob
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 117.0, 123.0))

    # Pass the blob through the network and detect faces
    face_net.setInput(blob)
    det = face_net.forward()
    detections = np.reshape(det, (200, 7))

    faces = []
    loc = []
    pred = []

    # Loop through detections
    for i in range(0, detections.shape[0]):
        # Extract probability
        confidence = detections[i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Compute x- and y-coordinates of bbox
            bbox = detections[i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = bbox.astype("int")

            # Ensure bbox is within dimensions of frame
            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

            # Extract RoI
            face = frame[start_y:end_y, start_x:end_x]
            # Convert extracted RoI to RGB
            cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # Resize RoI
            face = cv2.resize(face, (224, 224))
            # Convert PIL image to numpy array
            face = img_to_array(face)
            # Preprocess numpy array encoding a batch of images
            face = preprocess_input(face)

            # Add faces and associated bbox to their respective lists
            faces.append(face)
            loc.append((start_x, start_y, end_x, end_y))

    # Check if detection has been made
    if len(faces) > 0:
        # Convert list of faces to numpy array
        faces = np.array(faces, dtype="float32")

        # Classify faces as mask, no_mask, or incorrect_mask
        pred = mask_net.predict(faces, batch_size=32)

    return loc, pred


def determine_color(loc, pred):
    """Takes locations of classified faces and the corresponding predicted probabilities.
    Returns bounding box coordinates, color of bounding box and label, and the label"""

    # Loop over the detected face locations and their corresponding predictions
    for (bbox, confidence) in zip(loc, pred):
        # Unpack the bounding box and text
        (startX, startY, endX, endY) = bbox
        (incorrect_mask, mask, no_mask) = confidence

        # Determine the class label
        if mask > no_mask and mask > incorrect_mask:
            class_label = "Masked"
        elif incorrect_mask > mask and incorrect_mask > no_mask:
            class_label = "Incorrect Mask"
        else:
            class_label = "No Mask"

        # Determine the bounding box and accuracy text color
        if class_label == "Masked":
            clr = (0, 255, 0)  # Green
        elif class_label == "Incorrect Mask":
            clr = (0, 140, 255)  # Orange
        else:
            clr = (0, 0, 255)  # Red

        # Determine which probability is higher
        prob = max(mask, no_mask, incorrect_mask) * 100

        class_label = f"{class_label}: {prob:.2f}%"

        # Display the labels and bounding boxes on the output frame
        cv2.putText(frame, class_label, (startX, startY - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, clr, 1)
        cv2.rectangle(frame, (startX, startY), (endX, endY), clr, 2)


# Load the face detection model
face_net = cv2.dnn.readNet("Face Detectors/deploy.prototxt", 'Face Detectors/res10_300x300_ssd_iter_140000.caffemodel')

# Load the trained classifier model
classifier_model = load_model("classifier_model.model")

# Initialize video stream
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# capture = cv2.VideoCapture("filename.mp4")

# Loop over frames from video stream
while True:
    ret, frame = capture.read()

    frame = imutils.resize(frame, width=700)

    # Detect faces in the frame and classify
    locations, predictions = detect_mask(frame,
                                         face_net=face_net,
                                         mask_net=classifier_model)

    # Determine the color of bounding box and label text
    determine_color(locations, predictions)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Press the "Q"-key to break the while-loop and destroy windows
    if key == ord("q"):
        break

cv2.destroyAllWindows()
capture.release()
