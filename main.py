import cv2
import HandTrackingModule as htm    

wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam) 
cap.set(4, hCam) 

detector = htm.HandDetector(maxHands=2, detectionCon=0.7)

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    if detector.results.multi_hand_landmarks:
        number_of_hands = len(detector.results.multi_hand_landmarks)
        
        for i in range(number_of_hands):
            lmList = detector.findPosition(img, handNo=i)
            
            handLabel = detector.getHandLabel(i)
            
            if lmList and handLabel:
                x_list = [lm[1] for lm in lmList]
                y_list = [lm[2] for lm in lmList]
                
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                
                cv2.rectangle(img, (x_min-20, y_min-20), (x_max + 20, y_max+20), (255,192,203), 2)
                count, names, side = detector.analyzeHand(lmList, handLabel)
                text_x = x_min - 20
                text_y = y_min - 30 
                cv2.rectangle(img, (text_x, text_y - 60), (text_x + 250, text_y + 10), (0, 0, 0), cv2.FILLED)
                
                cv2.putText(img, f"{handLabel} ({side})", (text_x, text_y - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(img, f"Fing numbers: {count}", (text_x, text_y - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                names_str = ", ".join(names)
                cv2.putText(img, names_str, (x_min - 20, y_max + 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    window_name = "Hand Detector"
    cv2.imshow(window_name, img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    except:
        break

cap.release()
cv2.destroyAllWindows()