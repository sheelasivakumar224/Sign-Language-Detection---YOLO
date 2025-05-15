import cv2
import streamlit as st
import numpy as np
import time  # To manage detection time delays
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model using Ultralytics
model = YOLO("best.pt")

# Set up Streamlit layout
st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="🤟",
    layout="wide"
)

# Initialize session state for detected signs, time, and buffer
if 'detected_word' not in st.session_state:
    st.session_state.detected_word = ''  # Initialize an empty string for the detected word
if 'letter_buffer' not in st.session_state:
    st.session_state.letter_buffer = ''  # Buffer to prevent repeating letters
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = 0  # Track time between detections
if 'reset_flag' not in st.session_state:
    st.session_state.reset_flag = False  # A flag to manage reset state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None  # To store uploaded image
if 'detected_signs_list' not in st.session_state:
    st.session_state.detected_signs_list = []  # List to store all detected signs from the image

# Set detection delay (time in seconds)
detection_delay = 1.0  # 1 second between the same letter being appended

# Streamlit UI
st.title('Sign Language Detection System')
st.sidebar.title("Real-time Detection")
st.sidebar.title('Settings')

# Slider for confidence threshold
confidence_threshold = st.sidebar.slider('Confidence Threshold', min_value=0.0, max_value=1.0, value=0.5)

# Add a reset button to clear only the detected word
reset_word = st.sidebar.button("Reset Detected Word")  # Store the button press event

# Check if reset button is pressed and update the reset flag
if reset_word:
    st.session_state.reset_flag = True

# Only reset the detected word if reset_flag is True
if st.session_state.reset_flag:
    st.session_state.detected_word = ''
    st.session_state.detected_signs_list = []  # Reset detected signs list
    st.session_state.reset_flag = False  # Reset the flag so it doesn't reset again on next run
    st.sidebar.success("Detected word has been reset!")

# Main content - Choose input method
use_webcam = st.checkbox('Use Webcam')
use_ipcam = st.checkbox('Use IP cam')

cap = None
stframe = st.empty()
detected_signs_display = st.sidebar.empty()  # Create an empty placeholder for detected word

# IP Camera and Webcam input
if use_ipcam:
    st.write("IP Camera Stream:")
    cap = cv2.VideoCapture("https://192.168.176.180:8080/video")
elif use_webcam:
    st.write("Webcam Detection:")
    cap = cv2.VideoCapture(0)
else:
    st.write("Upload an image for detection:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        # Store uploaded image in session state
        st.session_state.uploaded_image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1))
        st.image(st.session_state.uploaded_image, channels="RGB", use_column_width=True)

# Process video frames if a valid video source is provided
if cap is not None and cap.isOpened():
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (640, 480))

            # YOLOv8 Prediction
            results = model.predict(frame)

            current_time = time.time()  # Get the current time
            for result in results:
                for box in result.boxes:
                    if box.conf > confidence_threshold:
                        class_id = int(box.cls[0].item())
                        detected_sign = result.names[class_id]  # Get the detected sign

                        # Only append the letter if it is different from the last one and enough time has passed
                        if detected_sign != st.session_state.letter_buffer and (current_time - st.session_state.last_detection_time) > detection_delay:
                            st.session_state.letter_buffer = detected_sign  # Update the buffer with the new gesture
                            st.session_state.current_letter = detected_sign  # Update the current letter
                            
                            # Only append to the word if it's different and the word is under the character limit
                            if len(st.session_state.detected_word) < 50:
                                st.session_state.detected_word += st.session_state.current_letter
                                print(st.session_state.detected_word)  # Print for debugging

                            # Update the last detection time
                            st.session_state.last_detection_time = current_time

            # Update the placeholder for detected word
            detected_signs_display.text("Detected Word: " + st.session_state.detected_word)

            # Convert frame to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in Streamlit
            stframe.image(frame, channels='RGB', use_column_width=True)

        else:
            break

    # Release the video capture
    cap.release()
else:
    if st.session_state.uploaded_image is not None:
        # If an image was uploaded, you might want to add detection here
        uploaded_frame = cv2.resize(st.session_state.uploaded_image, (640, 480))
        results = model.predict(uploaded_frame)

        # Clear previous detections
        st.session_state.detected_signs_list = []

        for result in results:
            for box in result.boxes:
                if box.conf > confidence_threshold:
                    class_id = int(box.cls[0].item())
                    detected_sign = result.names[class_id]  # Get the detected sign
                    uploaded_frame=result.plot()
                    # Append the detected sign to the list
                    if detected_sign not in st.session_state.detected_signs_list:
                        st.session_state.detected_signs_list.append(detected_sign)

        # Display all detected signs
        detected_signs_display.text("Detected Signs: " + ', '.join(st.session_state.detected_signs_list))

        # Convert uploaded image to RGB for Streamlit
        uploaded_frame = cv2.cvtColor(uploaded_frame, cv2.COLOR_BGR2RGB)
        stframe.image(uploaded_frame, channels='RGB', use_column_width=True)
    else:
        st.write("No video source available or unable to open the video feed.")