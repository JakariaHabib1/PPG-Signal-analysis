# PPG-Signal-analysis
import cv2
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd

def get_average_color(frame):
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define the region of interest (ROI) for the forehead area
    roi = hsv_frame[100:300, 200:400]
    # Calculate the average color in the ROI
    average_color = cv2.mean(roi)
    return average_color

def compute_heart_rate(video_source=0):
    cap = cv2.VideoCapture(video_source)

    # Lists to store the intensity values over time
    intensities = []
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video capture

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the average color value of the ROI
        average_color = get_average_color(frame)
        intensity = average_color[2]  # Use the V channel (brightness)
        intensities.append(intensity)

        # Display live feed
        cv2.rectangle(frame, (200, 100), (400, 300), (0, 255, 0), 2)
        cv2.putText(frame, 'Capturing PPG Signal...', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('PPG Signal Capture', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Convert the intensity list to a NumPy array for processing
    intensities = np.array(intensities)

    # High-pass filter to remove noise
    b, a = scipy.signal.butter(1, 0.5, 'high', fs=fps)
    filtered_signal = scipy.signal.filtfilt(b, a, intensities)

    # Automatically find the peaks
    peaks, _ = scipy.signal.find_peaks(filtered_signal, height=0.5, distance=fps/2)  # Adjust height and distance if necessary

    # Calculate the heart rate
    number_of_peaks = len(peaks)
    duration_in_seconds = len(filtered_signal) / fps
    heart_rate = (number_of_peaks / duration_in_seconds) * 60  # Heart rate in beats per minute (BPM)

    print(f"Estimated Heart Rate: {heart_rate:.2f} BPM")

    # Plot the filtered signal and mark the detected peaks
    plt.figure(figsize=(10, 5))
    plt.plot(filtered_signal, label='Filtered PPG Signal')
    plt.plot(peaks, filtered_signal[peaks], 'ro', label='Detected Peaks')
    plt.title(f'Filtered PPG Signal - Estimated Heart Rate: {heart_rate:.2f} BPM')
    plt.xlabel('Time (frames)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()

    return heart_rate

def compute_blood_pressure(blood_pressure_file):
    # Load the blood pressure data from the CSV file
    data = pd.read_csv(blood_pressure_file)

    # For demonstration, we will simply take the latest readings
    latest_reading = data.iloc[-1]
    systolic = latest_reading['systolic']
    diastolic = latest_reading['diastolic']

    print(f"Latest Blood Pressure: Systolic: {systolic} mmHg, Diastolic: {diastolic} mmHg")
    return systolic, diastolic

if __name__ == '__main__':
    video_source = 0  # Change this if you have a different video source
    blood_pressure_file = 'blood_pressure_data.csv'  # Path to the blood pressure CSV file
    
    heart_rate = compute_heart_rate(video_source)
    systolic, diastolic = compute_blood_pressure(blood_pressure_file)
