import cv2, pyautogui, time
import mediapipe as mp
import numpy as np
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
last_call = time.perf_counter()
render_time_total = 0
total_calls = 0

def get_time(tag="", do_print = False):
    global last_call, render_time_total, total_calls
    cur_call = time.perf_counter()
    cur_time = round((cur_call-last_call) * 100)/100
    if do_print:
        print("".join([tag, " ", str(cur_time).zfill(3), "s"]))
    last_call = cur_call
    render_time_total += cur_time
    total_calls += 1

cur_r_eye_centroid_x, cur_r_eye_centroid_y = 0, 0

eyeball_min_x, eyeball_max_x, eyeball_min_y, eyeball_max_y = -0.02029406753453339, 0.010147208517247985, -0.008228669112378906, 0.007702179930426878


try:
    while True:
        get_time("frame render time")
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        landmark_points = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks

        frame_h, frame_w, _ = frame.shape
        if landmark_points:
            landmarks = landmark_points[0].landmark

            r_eye_landmarks = np.array(landmarks)[[277, 282, 283, 285, 286, 283, 295, 300, 329, 333, 334, 336, 337, 339, 340, 341, 342, 343, 346, 347, 348, 349]]
            new_r_eye_centroid_x, new_r_eye_centroid_y = 0, 0
            
            # TODO: Try to make this as close to the center of the right eye
            r_eye_indexes = [277, 282, 283, 285, 286, 283, 295, 300, 329, 333, 334, 336, 337, 339, 340, 341, 342, 343, 346, 347, 348, 349]
            r_eye_landmarks = np.array(landmarks)[r_eye_indexes]
            for id, landmark in enumerate(r_eye_landmarks):
                #cv2.circle(frame, (int(landmark.x * frame_w), int(landmark.y * frame_h)), 3, (0, 0, 255))
                #cv2.putText(frame, str(r_eye_indexes[id]), (int(landmark.x * frame_w), int(landmark.y * frame_h)), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1)
                new_r_eye_centroid_x += landmark.x
                new_r_eye_centroid_y += landmark.y
            
            new_r_eye_centroid_x /= len(r_eye_landmarks)
            new_r_eye_centroid_y /= len(r_eye_landmarks)
            #print("r_eye_centroid", new_r_eye_centroid_x, new_r_eye_centroid_y)

            cv2.circle(frame, (int(new_r_eye_centroid_x * frame_w), int(new_r_eye_centroid_y * frame_h)), 3, (255, 0, 0))
            #cv2.circle(frame, (int(cur_r_eye_centroid_x * frame_w), int(cur_r_eye_centroid_y * frame_h)), 3, (0, 255, 0))
            cur_r_eye_centroid_x, cur_r_eye_centroid_y = new_r_eye_centroid_x, new_r_eye_centroid_y

            # Right Eyeball, create bounding box
            for id, landmark in enumerate(landmarks[473:474]):
                cv2.circle(frame, (int(landmark.x * frame_w), int(landmark.y * frame_h)), 3, (0, 255, 0))

                r_eyeball_x = landmark.x - cur_r_eye_centroid_x
                r_eyeball_y = landmark.y - cur_r_eye_centroid_y
                #print("".join(["\r", str(r_eyeball_x), " ", str(r_eyeball_y)]))

                # Create Bounding Box
                if r_eyeball_x < eyeball_min_x:
                    eyeball_min_x = r_eyeball_x
                if r_eyeball_x > eyeball_max_x:
                    eyeball_max_x = r_eyeball_x
                if r_eyeball_y < eyeball_min_y:
                    eyeball_min_y = r_eyeball_y
                if r_eyeball_y > eyeball_max_y:
                    eyeball_max_y = r_eyeball_y
                 
            # Show Bounding Box   
            r_eye_box_start = (int((cur_r_eye_centroid_x + eyeball_min_x) * frame_w), int((cur_r_eye_centroid_y + eyeball_min_y) * frame_h))
            r_eye_box_end = (int((cur_r_eye_centroid_x + eyeball_max_x) * frame_w), int((cur_r_eye_centroid_y + eyeball_max_y) * frame_h))
            print("r_eye_box_start, r_eye_box_end", r_eye_box_start, r_eye_box_end)
            cv2.rectangle(frame, r_eye_box_start, r_eye_box_end, (0,255,0), 1)
            
            # Move Mouse
            # screen_x = screen_w * landmark.x
            # screen_y = screen_h * landmark.y
            # pyautogui.moveTo(screen_x, screen_y, _pause=False)
                    
            # l_eye = [landmarks[145], landmarks[159]]
            # for landmark in l_eye:
            #     x = int(landmark.x * frame_w)
            #     y = int(landmark.y * frame_h)
            #     cv2.circle(frame, (x, y), 3, (0, 255, 255))

            #Points Finder: Puts numbers on the locations, change min and max for region
            # min,max = 300,350
            # for id, landmark in enumerate(landmarks[min:max]):
            #     cv2.putText(frame, str(id+min), (int(landmark.x * frame_w), int(landmark.y * frame_h)), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1)
        else:
            print("eyeball_min_x, eyeball_max_x, eyeball_min_y, eyeball_max_y", eyeball_min_x, eyeball_max_x, eyeball_min_y, eyeball_max_y)
            print("end report total and avg:", str(render_time_total).zfill(3), str(render_time_total / total_calls).zfill(3))
            exit(0)
        cv2.imshow('Eye Controlled Mouse', frame)
        cv2.waitKey(1)
except Exception as e:
    print("eyeball_min_x, eyeball_max_x, eyeball_min_y, eyeball_max_y", eyeball_min_x, eyeball_max_x, eyeball_min_y, eyeball_max_y)
    print("end report total and avg:", str(render_time_total).zfill(3), str(render_time_total / total_calls).zfill(3))
    print("errored", e)
