from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN

model=load_model('C:/Users/SHREYASH/Downloads/detect_model.h5')
vs=cv2.VideoCapture(0)
det=MTCNN()
def a():
     while True:
          ret,frame=vs.read()

          face=det.detect_faces(frame)

          img = cv2.resize(frame,(300,300))
          x = image.img_to_array(img)
          x = np.expand_dims(x, axis=0)
          pred=model.predict(x)

          if pred>0.5:
               pred='tejas'
          else:
               pred='shreyash'
          
          for i in face:
               x,y,width,height=i['box']
               cv2.putText(frame, pred, (x,y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
               cv2.rectangle(frame,(x,y),(x+width,y+height),(0,0,255),2)
               croped_face=frame[y-20:y+height,x:x+width]
               frame=cv2.blur(frame,(33,33))
               frame[y-20:y+height,x:x+width]=croped_face
          
          cv2.imshow('face detection',frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
     vs.release() 
     cv2.destroyAllWindows()
a()
     
