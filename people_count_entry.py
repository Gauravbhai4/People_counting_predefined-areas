
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from trackertt import*

model=YOLO('yolov8s.pt')
# area1=[(312,388),(289,390),(474,469),(497,462)]

area2=[(279,392),(250,397),(423,477),(454,469)]
area1=[(303,389),(507,472),(488,476),(293,395)]
# area1=[]
# area2=[]

# area2=[]
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('peoplecount1.mp4')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 


tracker = Tracker()

people_entry={}
people_exiting={}
entry=set()
exiting=set()

# count=0
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    # count += 1
    # if count % 2 != 0:
    #     continue
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.boxes
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
           list.append([x1,y1,x2,y2])
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        result=cv2.pointPolygonTest(np.array(area2,np.int32),((x4,y4)),False)
    #    print(result)
        if result>=0:
            people_entry[id]=(x4,y4)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
        if id in people_entry:
            result1=cv2.pointPolygonTest(np.array(area1,np.int32),((x4,y4)),False)
            if result1>=0:
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                cv2.circle(frame,(x4,y4),5,(2550,255),-1)
                cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
                entry.add(id)

        #people exiting##

        result2=cv2.pointPolygonTest(np.array(area1,np.int32),((x4,y4)),False)
    #    print(result)
        if result2>=0:
            people_exiting[id]=(x4,y4)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)

        if id in people_exiting:
            result3=cv2.pointPolygonTest(np.array(area2,np.int32),((x4,y4)),False)
            if result3>=0:
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
                cv2.circle(frame,(x4,y4),5,(2550,255),-1)
                cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
                exiting.add(id)
   

        
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('1'),(504,471),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('2'),(466,485),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)
    # print(people_entry)
    i=len(entry)
    o=len(exiting)
    cv2.putText(frame,str(i),(60,80),cv2.FONT_HERSHEY_COMPLEX,(0.7),(0,0,255),2)
    cv2.putText(frame,str(o),(60,140),cv2.FONT_HERSHEY_COMPLEX,(0.7),(255,0,255),2)


    cv2.imshow("RGB", frame)
    if cv2.waitKey(0)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()