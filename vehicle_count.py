import cv2
import numpy as np

cap = cv2.VideoCapture('Sample_video.mp4')


algo = cv2.bgsegm.createBackgroundSubtractorMOG()

count_line_position = 550

min_rect_width = 70
min_rect_height = 70

offset = 6


def centre_point(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1

    return cx,cy


detect = []
counter = 0

while True:
    ret, frame1 = cap.read()
    gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),5)

    img_sub = algo.apply(blur)

    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatdata = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatdata = cv2.morphologyEx(dilatdata,cv2.MORPH_CLOSE,kernel)

    counterShape, h = cv2.findContours(dilatdata,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    #cv2.imshow('Detector',dilatdata)

    cv2.line(frame1,(22,count_line_position),(1260,count_line_position),(255,0,255),4)

    for (i,c) in enumerate(counterShape):
        (x,y,w,h)=cv2.boundingRect(c)
        validate_counter = (w >= min_rect_width) and (h >= min_rect_height)
        if not validate_counter:
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(frame1,"Vehicle : "+ str(counter),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        centre = centre_point(x,y,w,h)
        detect.append(centre)
        cv2.circle(frame1,centre,4,(0,0,255),-1)

        for (x,y) in detect:
            if  y < (count_line_position + offset) and  y > (count_line_position - offset):
                counter+=1
                cv2.line(frame1,(22,count_line_position),(1260,count_line_position),(255,0,0),5)
                detect.remove((x,y))
                print("Total Vehicle passes: "+str(counter))


    cv2.putText(frame1,"Total Vehicles : "+ str(counter),(420,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),4)

    cv2.imshow('Original Video',frame1)

    if cv2.waitKey(1)==13:
        break

cv2.destroyAllWindows()
cap.release()
