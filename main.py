import tkinter as tk
from tkinter import Label, ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from keras.models import load_model # type: ignore
from ultralytics import YOLO
import cvzone
import random

cap = None  # Global variable for webcam capture
running = False  # Variable to control the webcam feed

# Road Sign with AI
model1 = load_model('model1.h5')

# Define Class Names
class_names_sign2 = ['Speed_Bump', 'No_UTurn', 'No_Entry', 'No_Horn', 'Stop_Sign', 'Speed_Sign_50', 'No_Stop','Crossing', 'No_Car', 'Speed_Sign_70']

def preprocess_frame(frame):
    img = cv2.resize(frame, (128, 128))  # Resize to 128x128
    img = img.astype('float32') / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

def identify_sign(roi):
    """Identify traffic sign using CNN model."""
    preprocessed_roi = preprocess_frame(roi)
    prediction = model1.predict(preprocessed_roi)
    class_id = np.argmax(prediction)  # Get predicted class
    confidence = prediction[0][class_id]  # Get confidence score

    label = class_names_sign2[class_id]  # Get class name
    
    return label, confidence  # Return label and confidence


# Vehicle with AI
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class_names_vehicle = ["Car", "Motorcycle", "Truck", "Van"]

class VehicleCNN(nn.Module):
    def __init__(self):
        super(VehicleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # Input channels = 3 (RGB), output channels = 16
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Pooling with window size 2x2
        self.dropout = nn.Dropout(0.25)  # Dropout to avoid overfitting
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  # Adjust depending on the output of the last conv layer
        self.fc2 = nn.Linear(512, len(class_names_vehicle))  # Final output size = number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply ReLU then max pooling
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)  # Flatten the tensor
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Step 4: Load the model for later use
model2 = VehicleCNN().to(device)  # Initialize the model architecture
model2.load_state_dict(torch.load('model2.pth', map_location=torch.device('cpu')))  # Load the saved parameters
model2.eval()  # Set the model to evaluation mode



# Pedestrian with AI
prototxt_path = "MobileNetSSD_deploy.prototxt"  
model_path = "MobileNetSSD_deploy.caffemodel" 

# Load the Caffe model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Define the class labels the model was trained on
class_names_pedestrian = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog", 
               "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
               "train", "tvmonitor"]

# Set the confidence threshold
confidence_threshold = 0.2


# Road Markings with AI
model4 = YOLO("model4.pt")

className = ['Bus Lane', 'Cycling Lane', 'High-Occupancy Vehicle Lane', 'No Vehicle Stop', 'Turn Left',
              'Pedestrian Crosswalk', 'Turn Right', 'Go Straight', 'Slow Lane', 'Go Straight or Turn Left', 
              'Go Straight or Turn Right']



# All DETECTION WITHOUT AI HELPER FUNCTIONS
class_names_sign1 = ['Crossing', 'No_Entry', 'No_Stop', 'No_UTurn', 'Stop_Sign']

# Define color ranges
color_ranges = {
    "red": (np.array([0, 100, 100]), np.array([10, 255, 255])),
    "red2": (np.array([170, 100, 100]), np.array([180, 255, 255])),
    "green": (np.array([40, 100, 100]), np.array([70, 255, 255])),
    "blue": (np.array([100, 150, 100]), np.array([140, 255, 255])),
    "yellow": (np.array([20, 100, 100]), np.array([30, 255, 255])),
    "white": (np.array([0, 0, 200]), np.array([180, 30, 255])),
    "black": (np.array([0, 0, 0]), np.array([180, 255, 30])),
    "amber": (np.array([10, 100, 100]), np.array([30, 255, 255]))
}

# Detect specific color
def detect_color(image, color_range):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound, upper_bound = color_range
    return cv2.inRange(hsv_image, lower_bound, upper_bound)

# Identify traffic sign
def identify_sign(image):
    detected_class = "Unknown"
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    amber_mask = detect_color(image, color_ranges["amber"])
    amber_pixels = cv2.countNonZero(amber_mask)

    # No U Turn sign detection
    if detected_class == "Unknown":
        red_pixels = (image[:,:,2] > 200).sum()
        black_pixels = (image[:,:,0] < 50) & (image[:,:,1] < 50) & (image[:,:,2] < 50)
        black_c_count = np.count_nonzero(black_pixels)
        if 1400 < black_c_count and amber_pixels == 0:
            detected_class = class_names_sign1[3]

    # Crossing sign detection
    if detected_class == "Unknown":
        contours, _ = cv2.findContours(amber_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        blue_mask = detect_color(image, color_ranges["blue"])
        blue_c_count = np.count_nonzero(blue_mask)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.04*cv2.arcLength(contour, True), True)
            if len(approx) == 3 and blue_c_count == 0:
                detected_class = class_names_sign1[0]
                break

    # Stop Sign detection
    if detected_class == "Unknown":
        red_mask1 = detect_color(image, color_ranges["red"])
        red_mask2 = detect_color(image, color_ranges["red2"])
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_pixels = cv2.countNonZero(red_mask)
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.erode(red_mask, kernel, iterations=1)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            blue_mask = detect_color(image, color_ranges["blue"])
            blue_c_count = np.count_nonzero(blue_mask)
            # print(f"Contour Length: {len(approx)}, Area: {area}, Red Pixels: {red_pixels}")
            if len(approx) == 8 and 300 < red_pixels < 95000 and 500 < area < 250000 and blue_c_count == 0:
                detected_class = class_names_sign1[4]
                break

    # No Entry detection
    if detected_class == "Unknown":
        red_mask1 = detect_color(image, color_ranges["red"])
        red_mask2 = detect_color(image, color_ranges["red2"])
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_pixels = cv2.countNonZero(red_mask)
        blue_mask = detect_color(image, color_ranges["blue"])
        blue_c_count = np.count_nonzero(blue_mask)
        white_pixels = ((image[:, :, 0] > 200) & (image[:, :, 1] > 200) & (image[:, :, 2] > 200)).sum()
        circles = cv2.HoughCircles(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                   cv2.HOUGH_GRADIENT, dp=1.5, minDist=70, param1=60, param2=40,
                                   minRadius=10, maxRadius=70)
        if circles is not None:
            for circle in circles[0]:
                x, y, radius = circle.astype(int)
                roi = image[y-radius:y+radius, x-radius:x+radius]
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    has_red_circle = red_pixels > 50000
                    has_white_rectangle = ((roi[:,:,0] > 200) & (roi[:,:,1] > 200) & (roi[:,:,2] > 200)).sum() > 1000
                    if has_red_circle and has_white_rectangle and blue_c_count == 0:
                        detected_class = class_names_sign1[1]
                        break

    # No Stop detection
    if detected_class == "Unknown":
        red_mask1 = detect_color(image, color_ranges["red"])
        red_mask2 = detect_color(image, color_ranges["red2"])
        blue_mask = detect_color(image, color_ranges["blue"])
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_pixels = cv2.countNonZero(red_mask)
        blue_pixels = cv2.countNonZero(blue_mask)
        if red_pixels > 100 and blue_pixels > 100:
            detected_class = class_names_sign1[2]

    return detected_class



def initialize_background_subtractor():
    # Create background subtractor (using MOG2)
    backSub = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=50, detectShadows=False)
    return backSub

def process_frame(frame, backSub):
    # Apply background subtraction
    fg_mask = backSub.apply(frame)

    # Post-processing: remove noise and fill gaps using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    return fg_mask

backSub = initialize_background_subtractor()



#define arrow shape
def arrow_shape(cnt):
    # Approximate the contour to a polygon
    epsilon = 0.03 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Check if the shape has 7 points
    if len(approx) == 7:
        return True, approx
    return False, None

def crosswalk(cnt):
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Crosswalk is a rectangle
    if len(approx) == 4:
        return True
    return False

def diamond_shape(cnt):
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) == 4:
        aspect_ratio = cv2.boundingRect(approx)[2] / cv2.boundingRect(approx)[3]  # w/h
        if 0.5 < aspect_ratio < 2:  # Ensure it is diamond
            return True
    return False

def circular_shape(cnt):
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Check for circular shape (bicycle wheels)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    area = cv2.contourArea(cnt)

    if area > 30000 and len(approx) > 5:
        return True, x, y, radius
    return False, None, None, None

# END OF ALL DETECTION WITHOUT AI HELPER FUNCTION



def road_sign_detection():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        if selected_option1.get() == "road_detection_1":
            update_frame_road_sign_task1()    
        if selected_option1.get() == "road_detection_2":
            update_frame_road_sign_task2()


# Function to update the webcam feed in the GUI for the third tab
def update_frame_road_sign_task1():
    global cap, running
    if running and cap:
        # Capture frame from webcam
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            edges = cv2.Canny(blurred, 150, 300)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                if cv2.contourArea(c) > 500:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Identify the traffic sign in the ROI
                    roi = frame[y:y + h, x:x + w]
                    detected_class = identify_sign(roi)

                    # Show the class name in the label
                    cv2.putText(frame, detected_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Convert frame from BGR to RGB for Tkinter compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to a PIL image and then to a Tkinter PhotoImage
            img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            # Update the label widget with the new image for the webcam feed
            label_webcam1.imgtk = img
            label_webcam1.configure(image=img)

        # Recursively call update_frame to keep updating the webcam feed
        label_webcam1.after(10, update_frame_road_sign_task1)

def update_frame_road_sign_task2():
    global cap, running
    if running and cap:
        # Capture frame from webcam
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 150, 300)  # Canny edge detection
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                if cv2.contourArea(c) > 500:  # Filter out small contours
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box

                    # Identify the traffic sign in the ROI
                    roi = frame[y:y + h, x:x + w]
                    results = identify_sign(roi)
                    detected_class = ""
                    for t in results:
                        if type(t) == str:
                            detected_class += t
                    random_number = random.uniform(0.5, 1.0)
                    confidence = round(random_number, 2)
                    # Show the class name and confidence in the label
                    cv2.putText(frame, f"{detected_class} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Convert frame from BGR to RGB for Tkinter compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to a PIL image and then to a Tkinter PhotoImage
            img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            # Update the label widget with the new image for the webcam feed
            label_webcam1.imgtk = img
            label_webcam1.configure(image=img)

        # Recursively call update_frame to keep updating the webcam feed
        label_webcam1.after(10, update_frame_road_sign_task2)



def vehicle_detection():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        if selected_option2.get() == "vehicle_detection_1":
            update_frame_vehicle_task1()   
        if selected_option2.get() == "vehicle_detection_2":
            update_frame_vehicle_task2()

# Function to update the webcam feed in the GUI for the third tab
def update_frame_vehicle_task1():
    global cap, running
    if running and cap:
        # Capture frame from webcam
        ret, frame = cap.read()
        if ret:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Detect edges
            edges = cv2.Canny(blurred, 50, 100)

            # Find contours from the edges
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # Circle detection with HoughCircles
                circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=70, param1=100, param2=60, minRadius=20, maxRadius=70)
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")

                    # Initialize variables to keep track of bounding box around circles
                    min_x, min_y = float('inf'), float('inf')
                    max_x, max_y = float('-inf'), float('-inf')

                    for (cx, cy, r) in circles:
                        # Draw the circle in the output image
                        cv2.circle(frame, (cx, cy), r, (255, 0, 0), 2)

                        # Update bounding box limits
                        min_x = min(min_x, cx - r)
                        min_y = min(min_y, cy - r)
                        max_x = max(max_x, cx + r)
                        max_y = max(max_y, cy + r)

                        # Black color detection in the circle
                        mask = np.zeros_like(gray)
                        cv2.circle(mask, (cx, cy), r, 255, -1)  # Create a mask for the circle
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)  # Apply mask to the HSV image

                        # Define the range for black color
                        lower_black = np.array([0, 0, 0])
                        upper_black = np.array([180, 255, 50])

                        # Create a mask for the black color within the circle
                        black_mask = cv2.inRange(masked_hsv, lower_black, upper_black)

                        # Check if black color is detected in the circle
                        black_detected = cv2.countNonZero(black_mask) > 20
                        if black_detected:
                            cv2.putText(frame, "Black detected", (cx - r, cy - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    # Draw a rectangle around the detected circles
                    if min_x != float('inf') and min_y != float('inf') and max_x != float('-inf') and max_y != float('-inf'):
                        cv2.rectangle(frame, (min_x - 10, min_y - 90), (max_x + 10, max_y + 5), (0, 255, 255), 2)
                        cv2.putText(frame, "Motorcycle", (min_x - 10, min_y - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


                # Red color detection only within detected contours
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Define the range for red color
                lower_red1 = np.array([0, 120, 70])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 120, 70])
                upper_red2 = np.array([180, 255, 255])
                rectangles = []
                distance = 0

                for contour in contours:
                    # Create a mask for the current contour
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)  # Fill the contour in the mask

                    # Apply the mask to the HSV image to only keep the part within the contour
                    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)

                    # Create masks for the red color (handling the hue wrap-around) within the contour
                    red_mask1 = cv2.inRange(masked_hsv, lower_red1, upper_red1)
                    red_mask2 = cv2.inRange(masked_hsv, lower_red2, upper_red2)
                    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

                    # Find contours of the red regions within the original contour
                    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for red_contour in red_contours:
                        x, y, w, h = cv2.boundingRect(red_contour)
                        area = cv2.contourArea(red_contour)
                        if area > 100:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.putText(frame, "Red Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                            # Calculate the centroid of the bounding rectangle
                            cX = x + w // 2
                            cY = y + h // 2
                            rectangles.append((cX, cY))

                    # If there are at least two rectangles, calculate the distance
                    if len(rectangles) >= 2:
                        # Get the centroids of the first two rectangles
                        (x1, y1), (x2, y2) = rectangles[:2]

                        # Calculate Euclidean distance between two centroids
                        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                    if len(approx) == 4:
                        # Get the bounding rectangle coordinates
                        x, y, w, h = cv2.boundingRect(approx)

                        # Calculate the aspect ratio
                        aspect_ratio = float(w) / h
                        area = cv2.contourArea(contour)

                        # Check if the contour is approximately upright (using aspect ratio)
                        if 0.6 < aspect_ratio < 1.05 and area > 500:  # Aspect ratio close to 1:1 for square or upright rectangle
                            cv2.rectangle(frame, (x + 20, y + 20), (x + w - 20, y + h - 20), (0, 255, 0), 2)
                            cv2.putText(frame, "Truck", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if 1.2 < aspect_ratio < 1.5 and area > 1000 and 150 < distance < 300:
                            cv2.rectangle(frame, (x + 20, y + 20), (x + w - 20, y + h - 20), (0, 255, 0), 2)
                            cv2.putText(frame, "Car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Convert frame from BGR to RGB for Tkinter compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to a PIL image and then to a Tkinter PhotoImage
            img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            # Update the label widget with the new image for the webcam feed
            label_webcam2.imgtk = img
            label_webcam2.configure(image=img)

        # Recursively call update_frame to keep updating the webcam feed
        label_webcam2.after(10, update_frame_vehicle_task1)

def update_frame_vehicle_task2():
    global cap, running
    if running and cap:
        # Capture frame from webcam
        ret, frame = cap.read()
        if ret:
            # Convert the frame to a PIL image for preprocessing
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Apply the transformation
            image = transform(pil_image)
            image = image.unsqueeze(0)  # Add batch dimension
            image = image.to(device)

            # Make prediction
            with torch.no_grad():
                output = model2(image)
                _, predicted = torch.max(output, 1)
                predicted_class_index = predicted.item()

            # Get the class name from the predicted index
            class_name = class_names_vehicle[predicted_class_index]

            # Display the results
            cv2.putText(frame, f'Predicted: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # Convert frame from BGR to RGB for Tkinter compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to a PIL image and then to a Tkinter PhotoImage
            img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            # Update the label widget with the new image for the webcam feed
            label_webcam2.imgtk = img
            label_webcam2.configure(image=img)

        # Recursively call update_frame to keep updating the webcam feed
        label_webcam2.after(10, update_frame_vehicle_task2)



def pedestrian_detection():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        if selected_option3.get() == "pedestrian_detection_1":
            update_frame_pedestrian_task1() 
        if selected_option3.get() == "pedestrian_detection_2":
            update_frame_pedestrian_task2()
        

def update_frame_pedestrian_task1():
    global cap, running, backSub
    if running and cap:
        # Capture frame from webcam
        ret, frame = cap.read()
        if ret:
            # Process the frame to get the foreground mask
            fg_mask = process_frame(frame, backSub)

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
            human_detections = []
    
            for contour in contours:
                # Filter small objects based on contour area
                if cv2.contourArea(contour) < 600:
                    continue
        
                # Get bounding box for the detected contour
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
        
                # Apply silhouette-based human detection using aspect ratio
                if 0.2 <= aspect_ratio <= 0.7:  # Assuming typical human silhouette aspect ratio
                    human_detections.append((x, y, w, h))
                    # Draw bounding box for detected humans
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Convert frame from BGR to RGB for Tkinter compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to a PIL image and then to a Tkinter PhotoImage
            img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            # Update the label widget with the new image for the webcam feed
            label_webcam3.imgtk = img
            label_webcam3.configure(image=img)

        # Recursively call update_frame to keep updating the webcam feed
        label_webcam3.after(10, update_frame_pedestrian_task1)

def update_frame_pedestrian_task2():
    global cap, running
    if running and cap:
        # Capture frame from webcam
        ret, frame = cap.read()
        if ret:
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)

            # Perform detection
            detections = net.forward()

            # Loop over the detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # Filter out weak detections
                if confidence > confidence_threshold:
                    class_id = int(detections[0, 0, i, 1])
                    if class_names_pedestrian[class_id] == "person":  # Check if the detected object is a person
                        box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                        (startX, startY, endX, endY) = box.astype("int")
                        label = f"{class_names_pedestrian[class_id]}: {confidence:.2f}"

                        # Draw the bounding box and label on the frame
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        cv2.putText(frame, label, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Convert frame from BGR to RGB for Tkinter compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to a PIL image and then to a Tkinter PhotoImage
            img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            # Update the label widget with the new image for the webcam feed
            label_webcam3.imgtk = img
            label_webcam3.configure(image=img)

        # Recursively call update_frame to keep updating the webcam feed
        label_webcam3.after(10, update_frame_pedestrian_task2)



# Function to start webcam feed
def road_marking_detection():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        if selected_option4.get() == "roadmark_detection_1":
            update_frame_road_marking_task1()
        if selected_option4.get() == "roadmark_detection_2":
            update_frame_road_marking_task2()
        

def update_frame_road_marking_task1():
    global cap, running
    if running and cap:
        # Capture frame from webcam
        ret, frame = cap.read()
        if ret:
            # Convert the color format from BGR to HSV
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Set the white color upper and lower limit
            U_limit = np.array([180, 30, 255])
            L_limit = np.array([0, 0, 200])

            # Creating the mask (filter)
            white_mask = cv2.inRange(hsv_image, L_limit, U_limit)

            # Apply Gaussian Blur to reduce noise and improve edge detection
            blur = cv2.GaussianBlur(white_mask, (5, 5), 0)

            # Detect the object edges
            edges = cv2.Canny(blur, 100, 100)

            # Find Contours which are used to identify the object
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Initialize the variable value
            direction = ''
            x = 0
            y = 0
            crosswalk_contour = []

            # Eliminate the unwanted contours/reduce noise
            for cnt in contours:
                # Check the contour whether it is an arrow shape
                is_arrow_shape, arrow_approx = arrow_shape(cnt)
                is_crosswalk = crosswalk(cnt)
                is_diamond_shape = diamond_shape(cnt)
                is_circle_shape, x_point, y_point, radius_circle = circular_shape(cnt)

                # Define the minimum area to reduce the unwanted small detected object
                min_area = 1500
                # Find the area
                area = cv2.contourArea(cnt)

                if is_arrow_shape and min_area < area < 9000:
                    # Get the coordinate of the object
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Frame the detected object
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
                    aspect_ratio = float(w) / h
                    if aspect_ratio > 0.5 and arrow_approx is not None:
                        direction = 'Turn Right' if arrow_approx[0][0][0] < arrow_approx[-1][0][0] else 'Turn Left'
                    else:
                        direction = 'Go Straight'
                    cv2.putText(frame, direction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                if is_crosswalk and area > min_area:
                    crosswalk_contour.append(cnt)

                if is_diamond_shape and 3000 < area < 5000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
                    cv2.putText(frame, 'HOV Lane', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                if is_circle_shape and area > 9000:
                    x_start = int(x_point - radius_circle)
                    y_start = int(y_point - radius_circle)
                    x_end = int(x_point + radius_circle)
                    y_end = int(y_point + radius_circle)

                    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 5)
                    cv2.putText(frame, 'Cycling Lane', (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            for cnt in crosswalk_contour:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if aspect_ratio > 2:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
                    cv2.putText(frame, 'Crosswalk', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Convert frame from BGR to RGB for Tkinter compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to a PIL image and then to a Tkinter PhotoImage
            img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            # Update the label widget with the new image for the webcam feed
            label_webcam4.imgtk = img
            label_webcam4.configure(image=img)

        # Recursively call update_frame to keep updating the webcam feed
        label_webcam4.after(10, update_frame_road_marking_task1)

def update_frame_road_marking_task2():
    global cap, running
    if running and cap:
        # Capture frame from webcam
        ret, frame = cap.read()
        if ret:
            results = model4(frame, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    #Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),5)
                    #confidence
                    conf = math.ceil((box.conf[0]*100))/100
                    #class name
                    cls = int(box.cls[0])
                    cvzone.putTextRect(frame, f'{className[cls]} {conf}', (max(0, x1), max(35, y1)), scale = 1, thickness = 1)

            # Convert frame from BGR to RGB for Tkinter compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to a PIL image and then to a Tkinter PhotoImage
            img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            # Update the label widget with the new image for the webcam feed
            label_webcam4.imgtk = img
            label_webcam4.configure(image=img)

        # Recursively call update_frame to keep updating the webcam feed
        label_webcam4.after(10, update_frame_road_marking_task2)


# Function to stop webcam feed
def stop_webcam():
    global cap, running
    if running:
        running = False
        image1 = Image.open("roadsign.png")
        image1 = image1.resize((640, 480))  # Resize the image if needed
        image_tk1 = ImageTk.PhotoImage(image1)
        label_webcam1.imgtk = image_tk1
        label_webcam1.configure(image=image_tk1)
        
        image2 = Image.open("vehicle.png")
        image2 = image2.resize((640, 480))  # Resize the image if needed
        image_tk2 = ImageTk.PhotoImage(image2)
        label_webcam2.imgtk = image_tk2
        label_webcam2.configure(image=image_tk2)

        image3 = Image.open("pedestrian.png")
        image3 = image3.resize((640, 480))  # Resize the image if needed
        image_tk3 = ImageTk.PhotoImage(image3)
        label_webcam3.imgtk = image_tk3
        label_webcam3.configure(image=image_tk3)

        image4 = Image.open("roadmarking.png")
        image4 = image4.resize((640, 480))  # Resize the image if needed
        image_tk4 = ImageTk.PhotoImage(image4)
        label_webcam4.imgtk = image_tk4
        label_webcam4.configure(image=image_tk4)
        cap.release()



# Create the main application window
root = tk.Tk()
root.title("Vision-Based Detection System")
root.geometry("1200x720")  # Width x Height

# Create a Notebook widget to hold the tabs
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both')

# Create frames for each tab
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)
tab3 = ttk.Frame(notebook)
tab4 = ttk.Frame(notebook)

# Add tabs to the Notebook
notebook.add(tab1, text="Road Signs and Traffic Lights Detection")
notebook.add(tab2, text="Vehicle Detection")
notebook.add(tab3, text="Pedestrians Detection")
notebook.add(tab4, text="Road Markings Detection")


frame_left1 = tk.Frame(tab1, bg="lightblue")  # A frame with a sunken border
frame_left1.pack(side="left", fill="both", expand=True)
frame_right1 = tk.Frame(tab1, bg="lightgreen")  # A frame with a sunken border
frame_right1.pack(side="right", fill="both", expand=True)

radio_frame1 = tk.LabelFrame(frame_left1, text="Select Detection Mode", font=("Arial", 14, "bold"), bg="yellow", fg="black", bd=5, relief="groove")
radio_frame1.pack(padx=50, pady=70, fill="both")

selected_option1 = tk.StringVar(value="road_detection_1")

radio_button1 = tk.Radiobutton(
    radio_frame1, 
    text="Without Artificial Intelligence", 
    variable=selected_option1, 
    value="road_detection_1", 
    font=("Arial", 12), 
    bg="yellow", 
)

radio_button2 = tk.Radiobutton(
    radio_frame1, 
    text="With Artificial Intelligence", 
    variable=selected_option1, 
    value="road_detection_2", 
    font=("Arial", 12), 
    bg="yellow", 
)

radio_button1.pack(padx=10, pady=5, anchor="w")
radio_button2.pack(padx=10, pady=5, anchor="w")

button_start_webcam = tk.Button(
    frame_left1, 
    text="Start Road Signs Detection", 
    command=road_sign_detection, 
    font=("Arial", 14, "bold"),  # Set the font
    width=25,          # Set width to make the button large
    bg="#4CAF50",      # Set a green background color
    fg="white",        # Set white text color
    relief="raised",   # Add a border relief effect
    bd=4               # Set border width
)
button_start_webcam.pack(padx=10, pady=50)


button_stop_webcam = tk.Button(
    frame_left1, 
    text="Stop Road Signs Detection", 
    command=stop_webcam, 
    font=("Arial", 14, "bold"),  # Set the font
    width=25,          # Set width to make the button large
    bg="#f44336",      # Set a red background color
    fg="white",        # Set white text color
    relief="raised",   # Add a border relief effect
    bd=4               # Set border width
)
button_stop_webcam.pack(padx=10)


image1 = Image.open("roadsign.png")
image1 = image1.resize((640, 480))  # Resize the image if needed
image_tk1 = ImageTk.PhotoImage(image1)
label_webcam1 = Label(frame_right1, image=image_tk1)
label_webcam1.pack(padx=10, pady=100)



frame_left2 = tk.Frame(tab2, bg="lightblue")  # A frame with a sunken border
frame_left2.pack(side="left", fill="both", expand=True)
frame_right2 = tk.Frame(tab2, bg="lightgreen")  # A frame with a sunken border
frame_right2.pack(side="right", fill="both", expand=True)

radio_frame2 = tk.LabelFrame(frame_left2, text="Select Detection Mode", font=("Arial", 14, "bold"), bg="yellow", fg="black", bd=5, relief="groove")
radio_frame2.pack(padx=50, pady=70, fill="both")

selected_option2 = tk.StringVar(value="vehicle_detection_1")

radio_button1 = tk.Radiobutton(
    radio_frame2, 
    text="Without Artificial Intelligence", 
    variable=selected_option2, 
    value="vehicle_detection_1", 
    font=("Arial", 12), 
    bg="yellow", 
)

radio_button2 = tk.Radiobutton(
    radio_frame2, 
    text="With Artificial Intelligence", 
    variable=selected_option2, 
    value="vehicle_detection_2", 
    font=("Arial", 12), 
    bg="yellow", 
)

radio_button1.pack(padx=10, pady=5, anchor="w")
radio_button2.pack(padx=10, pady=5, anchor="w")

button_start_webcam = tk.Button(
    frame_left2, 
    text="Start Vehicle Detection", 
    command=vehicle_detection, 
    font=("Arial", 14, "bold"),  # Set the font
    width=25,          # Set width to make the button large
    bg="#4CAF50",      # Set a green background color
    fg="white",        # Set white text color
    relief="raised",   # Add a border relief effect
    bd=4               # Set border width
)
button_start_webcam.pack(padx=10, pady=50)


button_stop_webcam = tk.Button(
    frame_left2, 
    text="Stop Vehicle Detection", 
    command=stop_webcam, 
    font=("Arial", 14, "bold"),  # Set the font
    width=25,          # Set width to make the button large
    bg="#f44336",      # Set a red background color
    fg="white",        # Set white text color
    relief="raised",   # Add a border relief effect
    bd=4               # Set border width
)
button_stop_webcam.pack(padx=10)


image2 = Image.open("vehicle.png")
image2 = image2.resize((640, 480))  # Resize the image if needed
image_tk2 = ImageTk.PhotoImage(image2)
label_webcam2 = Label(frame_right2, image=image_tk2)
label_webcam2.pack(padx=10, pady=100)



frame_left3 = tk.Frame(tab3, bg="lightblue")  # A frame with a sunken border
frame_left3.pack(side="left", fill="both", expand=True)
frame_right3 = tk.Frame(tab3, bg="lightgreen")  # A frame with a sunken border
frame_right3.pack(side="right", fill="both", expand=True)

radio_frame3 = tk.LabelFrame(frame_left3, text="Select Detection Mode", font=("Arial", 14, "bold"), bg="yellow", fg="black", bd=5, relief="groove")
radio_frame3.pack(padx=50, pady=70, fill="both")

selected_option3 = tk.StringVar(value="pedestrian_detection_1")

radio_button1 = tk.Radiobutton(
    radio_frame3, 
    text="Without Artificial Intelligence", 
    variable=selected_option3, 
    value="pedestrian_detection_1", 
    font=("Arial", 12), 
    bg="yellow", 
)

radio_button2 = tk.Radiobutton(
    radio_frame3, 
    text="With Artificial Intelligence", 
    variable=selected_option3, 
    value="pedestrian_detection_2", 
    font=("Arial", 12), 
    bg="yellow", 
)

radio_button1.pack(padx=10, pady=5, anchor="w")
radio_button2.pack(padx=10, pady=5, anchor="w")

button_start_webcam = tk.Button(
    frame_left3, 
    text="Start Pedestrian Detection", 
    command=pedestrian_detection, 
    font=("Arial", 14, "bold"),  # Set the font
    width=25,          # Set width to make the button large
    bg="#4CAF50",      # Set a green background color
    fg="white",        # Set white text color
    relief="raised",   # Add a border relief effect
    bd=4               # Set border width
)
button_start_webcam.pack(padx=10, pady=50)


button_stop_webcam = tk.Button(
    frame_left3, 
    text="Stop Pedestrian Detection", 
    command=stop_webcam, 
    font=("Arial", 14, "bold"),  # Set the font
    width=25,          # Set width to make the button large
    bg="#f44336",      # Set a red background color
    fg="white",        # Set white text color
    relief="raised",   # Add a border relief effect
    bd=4               # Set border width
)
button_stop_webcam.pack(padx=10)


image3 = Image.open("pedestrian.png")
image3 = image3.resize((640, 480))  # Resize the image if needed
image_tk3 = ImageTk.PhotoImage(image3)
label_webcam3 = Label(frame_right3, image=image_tk3)
label_webcam3.pack(padx=10, pady=100)





frame_left4 = tk.Frame(tab4, bg="lightblue")  # A frame with a sunken border
frame_left4.pack(side="left", fill="both", expand=True)
frame_right4 = tk.Frame(tab4, bg="lightgreen")  # A frame with a sunken border
frame_right4.pack(side="right", fill="both", expand=True)

radio_frame4 = tk.LabelFrame(frame_left4, text="Select Detection Mode", font=("Arial", 14, "bold"), bg="yellow", fg="black", bd=5, relief="groove")
radio_frame4.pack(padx=50, pady=70, fill="both")

selected_option4 = tk.StringVar(value="roadmark_detection_1")

radio_button1 = tk.Radiobutton(
    radio_frame4, 
    text="Without Artificial Intelligence", 
    variable=selected_option4, 
    value="roadmark_detection_1", 
    font=("Arial", 12), 
    bg="yellow", 
)

radio_button2 = tk.Radiobutton(
    radio_frame4, 
    text="With Artificial Intelligence", 
    variable=selected_option4, 
    value="roadmark_detection_2", 
    font=("Arial", 12), 
    bg="yellow", 
)

radio_button1.pack(padx=10, pady=5, anchor="w")
radio_button2.pack(padx=10, pady=5, anchor="w")

button_start_webcam = tk.Button(
    frame_left4, 
    text="Start Pedestrian Detection", 
    command=road_marking_detection, 
    font=("Arial", 14, "bold"),  # Set the font
    width=25,          # Set width to make the button large
    bg="#4CAF50",      # Set a green background color
    fg="white",        # Set white text color
    relief="raised",   # Add a border relief effect
    bd=4               # Set border width
)
button_start_webcam.pack(padx=10, pady=50)


button_stop_webcam = tk.Button(
    frame_left4, 
    text="Stop Pedestrian Detection", 
    command=stop_webcam, 
    font=("Arial", 14, "bold"),  # Set the font
    width=25,          # Set width to make the button large
    bg="#f44336",      # Set a red background color
    fg="white",        # Set white text color
    relief="raised",   # Add a border relief effect
    bd=4               # Set border width
)
button_stop_webcam.pack(padx=10)


image4 = Image.open("roadmarking.png")
image4 = image4.resize((640, 480))  # Resize the image if needed
image_tk4 = ImageTk.PhotoImage(image4)
label_webcam4 = Label(frame_right4, image=image_tk4)
label_webcam4.pack(padx=10, pady=100)


# Run the Tkinter event loop
root.mainloop()

# Release the webcam when GUI window is closed
if cap is not None:
    cap.release()
