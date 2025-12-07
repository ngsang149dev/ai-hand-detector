import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands, 
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.finger_names = ["Thumb", "Index_finger", "Middle_finger", "Ring_finger", "Pinky"]
        self.tip_ids = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
        return lmList

    def getHandLabel(self, index):
        if self.results.multi_handedness:
            if index < len(self.results.multi_handedness):
                classification = self.results.multi_handedness[index].classification[0]
                return classification.label
        return None

    def analyzeHand(self, lmList, handLabel):
        fingers_up = []
        fingers_status = []
        palm_side = "Undetected"

        thumb_x = lmList[4][1]
        pinky_x = lmList[20][1]

        if handLabel == "Right":
            palm_side = "Palm" if thumb_x > pinky_x else "Back"
        else:
            palm_side = "Palm" if thumb_x < pinky_x else "Back"


        is_thumb_up = False
        if handLabel == "Right":
            if palm_side == "Palm":
                if lmList[4][1] > lmList[3][1]: is_thumb_up = True
            else: # Back
                if lmList[4][1] < lmList[3][1]: is_thumb_up = True
        else: # Left
            if palm_side == "Palm":
                if lmList[4][1] < lmList[3][1]: is_thumb_up = True
            else: # Back
                if lmList[4][1] > lmList[3][1]: is_thumb_up = True
        
        if is_thumb_up:
            fingers_up.append(self.finger_names[0])
            fingers_status.append(1)
        else:
            fingers_status.append(0)

        for id in range(1, 5):
            if lmList[self.tip_ids[id]][2] < lmList[self.tip_ids[id] - 2][2]:
                fingers_up.append(self.finger_names[id])
                fingers_status.append(1)
            else:
                fingers_status.append(0)

        return fingers_status.count(1), fingers_up, palm_side