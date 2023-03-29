from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import vg

import mido

from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

import emm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# # For static images:
# IMAGE_FILES = []
# with mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=2,
#     min_detection_confidence=0.5) as hands:
#   for idx, file in enumerate(IMAGE_FILES):
#     # Read an image, flip it around y-axis for correct handedness output (see
#     # above).
#     image = cv2.flip(cv2.imread(file), 1)
#     # Convert the BGR image to RGB before processing.
#     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # Print handedness and draw hand landmarks on the image.
#     print('Handedness:', results.multi_handedness)
#     if not results.multi_hand_landmarks:
#       continue
#     image_height, image_width, _ = image.shape
#     annotated_image = image.copy()
#     for hand_landmarks in results.multi_hand_landmarks:
#       print('hand_landmarks:', hand_landmarks)
#       print(
#           f'Index finger tip coordinates: (',
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#       )
#       mp_drawing.draw_landmarks(
#           annotated_image,
#           hand_landmarks,
#           mp_hands.HAND_CONNECTIONS,
#           mp_drawing_styles.get_default_hand_landmarks_style(),
#           mp_drawing_styles.get_default_hand_connections_style())
#     cv2.imwrite(
#         '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
#     # Draw hand world landmarks.
#     if not results.multi_hand_world_landmarks:
#       continue
#     for hand_world_landmarks in results.multi_hand_world_landmarks:
#       mp_drawing.plot_landmarks(
#         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

def finger_angles(sk: list[np.ndarray], start: int):
    finger_1 = sk[start] - sk[0]
    finger_2 = sk[start + 1] - sk[start]
    finger_3 = sk[start + 2] - sk[start + 1]
    finger_4 = sk[start + 3] - sk[start + 2]

    return vg.angle(finger_1, finger_2), vg.angle(finger_2, finger_3), vg.angle(finger_3, finger_4)


def fingers_angles(sk: list[np.ndarray]):
    finger_0_angle = sum(finger_angles(sk, 1))
    finger_1_angle = sum(finger_angles(sk, 5))
    finger_2_angle = sum(finger_angles(sk, 9))
    finger_3_angle = sum(finger_angles(sk, 13))
    finger_4_angle = sum(finger_angles(sk, 17))

    return finger_0_angle, finger_1_angle, finger_2_angle, finger_3_angle, finger_4_angle


def fingers_score(sk: list[np.ndarray]):
    return tuple(map(lambda score: (score - 60) / 140, fingers_angles(sk)))


def open_close_score(sk: list[np.ndarray]):
    """
    从手部关键点世界坐标计算开闭程度
    :param sk: 在世界坐标系下的手部关键点列表，长度为 21，元素为 3 维向量
    :return: 输出为 0~1 之间的数值，越接近 1 说明越开
    """

    fs = fingers_score(sk)
    print(fs)
    print(sum(fs))
    score = sum(fs) / 5
    # unified_score = max(0, min(1, (score - 60) / 140))
    return max(0, min(1, score))


serial_spin_should_stop = False

from threading import Thread, Condition


def serial_spin(que: deque[float], cond: Condition):
    from emm import Emm
    emm.do_log = False
    motor = Emm("/dev/ttyS0")

    direction = False
    import time

    last_invert = time.time()

    while not serial_spin_should_stop:
        with cond:
            cond.wait()
        if len(que) == 0:
            continue
        que_copy = que.copy()
        que.clear()
        que_sum = sum(que_copy)
        que_len = len(que_copy)
        average = que_sum / que_len
        result = int(average * 36)
        print(f"sum: {que_sum}, len: {que_len}, average: {average}, result: {result}")
        try:
            if motor.read_stuck():
                # direction *= -1
                now = time.time()
                dt = now - last_invert
                if dt > 5:
                    print(f"Stucked, try inverting ({dt} since last)")
                    direction = not direction
                    last_invert = now
                else:
                    print(f"Stucked, last invert try in {dt}")
            motor.set_speed_mode(direction, result, 64)
        except IndexError as e:
            print(f"Countered `IndexError` ({e}), motor maybe offline?")


que = deque[float]()
cond = Condition()

serial_thread = Thread(target=serial_spin, args=(que, cond,))
serial_thread.start()

# from emm import Emm
# emm.do_log = False
#
# motor = Emm("/dev/ttyUSB0")

# For webcam input:
cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)

midi_out = mido.open_output()

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            marked_hands = results.multi_hand_world_landmarks
            marked_hands: list[NormalizedLandmarkList]

            sk = [np.array([lm.x, lm.y, lm.z]) for lm in marked_hands[0].landmark]

            fs = fingers_score(sk)
            for i in range(5):
                value = int(max(0, min(127, fs[i]*128)))
                midi_out.send(mido.Message("control_change", control=i+1, value=value))
            # print(fs)
            # print(sum(fs))
            score = sum(fs) / 5
            # unified_score = max(0, min(1, (score - 60) / 140))
            que.append(max(0, min(1, score)) ** 0.8)
            with cond:
                cond.notify()

            # print(results.multi_hand_landmarks[0].landmark[0])
            # print(dir(results.multi_hand_landmarks[0]))
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break
        if cv2.pollKey() == ord("q"):
            break
cap.release()
serial_spin_should_stop = True
with cond:
    cond.notify()
serial_thread.join()
