import cv2
from playsound import playsound
import mediapipe as mp
import time

HIGH_VALUE = 10000
WIDTH = HIGH_VALUE
HEIGHT = HIGH_VALUE


cap= cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

pTime=0
cTime=0

while True:
    _,img=cap.read()
    img=cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    cv2.rectangle(img, (0,0), (95,250), (255,255,255),1)
    cv2.rectangle(img, (95,0), (190,250), (255,255,255),1)
    cv2.rectangle(img, (190,0), (285,250), (255,255,255),1)
    cv2.rectangle(img, (285,0), (380,250), (255,255,255),1)
    cv2.rectangle(img, (380,0), (475,250), (255,255,255),1)
    cv2.rectangle(img, (475,0), (570,250), (255,255,255),1)
    cv2.rectangle(img, (570,0), (665,250), (255,255,255),1)
    cv2.rectangle(img, (665,0), (760,250), (255,255,255),1)
    cv2.rectangle(img, (760,0), (855,250), (255,255,255),1)
    cv2.rectangle(img, (855,0), (950,250), (255,255,255),1)
    cv2.rectangle(img, (950,0), (1045,250), (255,255,255),1)
    cv2.rectangle(img, (1045,0), (1140,250), (255,255,255),1)
    
    cv2.putText(img, "C", (47, 170), cv2.FONT_HERSHEY_PLAIN, 1,(255, 225, 255))
    cv2.putText(img, "C#", (142, 170), cv2.FONT_HERSHEY_PLAIN, 1,(255, 225, 255))
    cv2.putText(img, 'D', (237, 170), cv2.FONT_HERSHEY_PLAIN, 1,(255, 225, 255))
    cv2.putText(img, "D#", (332, 170), cv2.FONT_HERSHEY_PLAIN, 1,(255, 225, 255))
    cv2.putText(img, "E", (427, 170), cv2.FONT_HERSHEY_PLAIN, 1,(255, 225, 255))
    cv2.putText(img, "F", (522, 170), cv2.FONT_HERSHEY_PLAIN, 1,(255, 225, 255))
    cv2.putText(img, "F#", (617, 170), cv2.FONT_HERSHEY_PLAIN, 1,(255, 225, 255))
    cv2.putText(img, "G", (712, 170), cv2.FONT_HERSHEY_PLAIN, 1,(255, 225, 255))
    cv2.putText(img, "G#", (807, 170), cv2.FONT_HERSHEY_PLAIN, 1,(255, 225, 255))
    cv2.putText(img, "A", (902, 170), cv2.FONT_HERSHEY_PLAIN, 1,(255, 225, 255))
    cv2.putText(img, "B", (997, 170), cv2.FONT_HERSHEY_PLAIN, 1,(255, 225, 255))
    cv2.putText(img, "C", (1092, 170), cv2.FONT_HERSHEY_PLAIN, 1,(255, 225, 255))
    
    
    results=hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                
                if id in[4,8,12,16,20]:
                    if cx > 0 and cy > 0 and cx < 95 and cy < 250:
                        playsound('notes/C.wav',block=False)
                        break      
                    if cx > 95 and cy > 0  and cx < 190 and cy < 250:
                        playsound('notes/C_s.wav',block=False) 
                        break      
                    if cx > 190 and cy > 0 and cx < 285 and cy < 250:
                        playsound('notes/D.wav',block=False) 
                        break      
                    if cx > 285 and cy > 0 and cx < 380 and cy < 250:
                        playsound('notes/D_s.wav',block=False) 
                        break      
                    if cx > 380 and cy > 0 and cx < 475 and cy < 250:
                        playsound('notes/E.wav',block=False)
                        break      
                    if cx > 475 and cy > 0  and cx < 570 and cy < 250:
                        playsound('notes/F.wav',block=False)
                        break      
                    if cx > 570 and cy > 0 and cx < 665 and cy < 250:
                        playsound('notes/F_s.wav',block=False)
                        break      
                    if cx > 665 and cy > 0 and cx < 760 and cy < 250:
                        playsound('notes/G.wav',block=False)
                        break
                    if cx > 760 and cy > 0 and cx < 855 and cy < 250:
                        playsound('notes/G_s.wav',block=False)
                        break      
                    if cx > 855 and cy > 0  and cx < 950 and cy < 250:
                        playsound('notes/A.wav',block=False)
                        break      
                    if cx > 950 and cy > 0 and cx < 1045 and cy < 250:
                        playsound('notes/B.wav',block=False)
                        break      
                    if cx > 1045 and cy > 0 and cx < 1140 and cy < 250:
                        playsound('notes/C1.wav',block=False)
                        break 
                
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
            
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)
    
    cv2.imshow('Image',img)
    
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()