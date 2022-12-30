# Imports
import cv2
import mediapipe as mp

#-------------------------------------
# Setup
mp_drawing = mp.solutions.drawing_utils # for drawing on screen
mp_drawing_styles = mp.solutions.drawing_styles # drawing style
mp_pose = mp.solutions.pose # get pose estimation

#-------------------------------------
# Webcam Input

cap = cv2.VideoCapture(0) # 0 for built-in cam || 1,2,.. for external camera
with mp_pose.Pose(
    min_detection_confidence = 0.6, # minimum percentage for decections
    min_tracking_confidence = 0.6
    ) as pose :
    while cap.isOpened() :
        ret, frame = cap.read()
        if not ret:
            print("Ignoring No Video in Camera frame")
            continue

#----------------------------------------
# Drawing
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert to RGB
        results = pose.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # convert to BGR

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imshow("Pose output", cv2.flip(frame, 1))

        if cv2.waitKey(5) & 0xFF == ord('q'): # hit q to stop
            break

cap.release()