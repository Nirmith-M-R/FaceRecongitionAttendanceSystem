import cv2
import face_recognition
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import os


class Attendance:
    def __init__(self):
        self.load_images = []
        self.Image_locations = []
        self.ImageName = []
        self.Img_array = []
        self.encode_images = []
        self.images = os.listdir("FlaskApp/photos")
        for i in self.images:
            self.load_images.append(face_recognition.load_image_file(f'FlaskApp/photos/{i}'))
            self.ImageName.append(i)
            #self.Img_array.append(np.array(self.load_images[-1]))
            self.Image_locations.append(face_recognition.face_locations(self.load_images[-1]))
            self.encode_images.append(face_recognition.face_encodings(self.load_images[-1], self.Image_locations[-1])[0])
        self.face_locations = []
        self.face_encodings = []
        self.name = []

    def Picture(self, img, resizeRange=5):
        try:
            frame = Image.open(img)
        except:
            frame = img
        self.frame = np.array(frame)
        self.resizeFrame = cv2.resize(self.frame, (0, 0), fx=resizeRange, fy=resizeRange)
        self.rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.face_locations = face_recognition.face_locations(self.rgb_frame)
        self.face_encodings = face_recognition.face_encodings(self.rgb_frame, self.face_locations)
        for encoding in self.face_encodings:
            self.matches = face_recognition.compare_faces(self.encode_images, encoding, tolerance=0.6)
            self.face_distance = face_recognition.face_distance(self.encode_images, encoding)
            #print(self.ImageName, "\n", self.matches,"\n",self.face_locations,"\n", self.Image_locations)
            self.best_match_index = np.argmin(self.face_distance)
            if self.matches[self.best_match_index]:
                self.name.append(self.ImageName[self.best_match_index])

    def Video(self, video):
        cap = cv2.VideoCapture(video)
        success = 1

        while success:
            success, self.frame = cap.read()
            if success:
                self.rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.face_locations = face_recognition.face_locations(self.rgb_frame)
                self.face_encodings = face_recognition.face_encodings(self.rgb_frame, self.face_locations)
                for encoding in self.face_encodings:
                    self.matches = face_recognition.compare_faces(self.encode_images, encoding, tolerance=0.6)
                    self.face_distance = face_recognition.face_distance(self.encode_images, encoding)
                    #print(self.ImageName, "\n", self.matches,"\n",self.face_locations,"\n", self.Image_locations)
                    self.best_match_index = np.argmin(self.face_distance)
                    if self.matches[self.best_match_index]:
                        self.name.append(self.ImageName[self.best_match_index])
            else:
                break

    def csvFile(self):
        names = []
        for i in self.name:
            names.append(i.split('.')[0])
        df = pd.DataFrame()
        name = set(self.name)
        name = list(name)
        pdseries = pd.Series(name)
        df.insert(0, 'Roll:', value=pdseries, allow_duplicates=True)
        #df.to_csv(f'{datetime.today()}.csv')
        return df


if __name__ == '__main__':
    ob = Attendance()
    ob.Picture("FlaskApp/photos/group.jpg", resizeRange=1)
    names = []
    # ob.Video('video1.mp4')
    for i in ob.name:
        names.append(i.split('.')[0])
    print(names)
    df = ob.csvFile()
    print(df)
