import cv2
import boto3
import datetime
import requests
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

count=0

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        #count=0
        global count
        success, image = self.video.read()
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        image1 = im_buf_arr.tobytes()
        client=boto3.client('rekognition',
                        aws_access_key_id="ASIASQQKOUBEY3Y6NKPY",
                        aws_secret_access_key="mMU+aOe1c82vvvWMlyYjsQpvoOiMokcavxqYt2CC",
                        aws_session_token="FwoGZXIvYXdzEFEaDEhSh9ptGyzwS7ioLCLEARWF3WJ8OJWSWjzYxPni+S5GzBucRIBZVDRhNmk+8Pc3EjzdZUMM7jo8VVud8uxC0+2bnCn6G1Z8xv7UjknLz437WeMbJERQhN83EJl6pfzkGkwCHGelnLGOX+USRZgQ3vP4RYq9DuF8bITvDowzNuU+C6jm8rTwwxuUyipnHuxWG+gA0ZF4hbL67suVudnQoPLueKWrJUd0GQuvLHuyonJMrQ2VyB97YUOXPOxKm57yHIT07H/Mfct+H+E3JkEB3nQg/oUoxezj+gUyLRXqPhisqUtY3Cfl+/MwAsJ7RCbSYcLyLKJWgiWLSzHg/B3fTLiZ0+p5FdvOWg==",
                        region_name='us-east-1')
        response = client.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:us-east-1:172894363721:project/mask-detect/version/mask-detect.2020-09-09T02.25.34/1599598535019',Image={
            'Bytes':image1})
        print(response['CustomLabels'])

        if not len(response['CustomLabels']):
            count=count+1
            date = str(datetime.datetime.now()).split(" ")[0]
            #print(date)
            url = "https://awv33bix5c.execute-api.us-east-1.amazonaws.com/countmask/maskcount?date="+date+"&count="+str(count)
            resp = requests.get(url)
            f = open("countfile.txt", "w")
            f.write(str(count))
            f.close()
            #print(count)

        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)

        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
        	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        	break
        ret, jpeg = cv2.imencode('.jpg', image)
        #cv2.putText(image, text = str(count), org=(10,40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(1,0,0))
        cv2.imshow('image',image)
        return jpeg.tobytes()
