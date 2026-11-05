import argparse
import functools
import time
import cv2
import pyttsx3  #!!!语音
import threading    
 
from utils.utils_train import add_arguments, print_arguments
from infer import Predictor

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('camera_id',                int,     0,                                  '使用的相机ID')
add_arg('face_db_path',             str,     'images_db',                        '人脸库路径')
add_arg('threshold',                float,   0.6,                                '判断相识度的阈值')
add_arg('mobilefacenet_model_path', str,     'save_model/mobilefacenet.pth',     'MobileFaceNet预测模型的路径')
add_arg('mtcnn_model_path',         str,     'save_model/mtcnn',                 'MTCNN预测模型的路径')
args = parser.parse_args()
print_arguments(args)

###
checkin_book ={}   

### !!!语音
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # 设置语速，默认 200，数值越大语速越快
engine.setProperty('volume', 1)
def __speakthread(text):
    global engine
    engine.say(text)
    engine.runAndWait() 
def asyncSpeak(text) :
    thread = threading.Thread(target=__speakthread, args=(text,));thread.start()
   




if __name__ == '__main__':
    
    #asyncSpeak("程序启动")    
     
    predictor = Predictor(args.mtcnn_model_path, args.mobilefacenet_model_path, args.face_db_path, threshold=args.threshold)
    cap = cv2.VideoCapture(args.camera_id)
    while True:
        ret, image = cap.read()
        if not ret:
            print("获取图像失败！")
            break
        boxes, names = predictor.recognition(image)
        if names:
           for name in names:
               if predictor.get_checkin(name)==0:
                     #asyncSpeak(name+"签到成功")
                     predictor.set_checkin(name)                 
        image = predictor.draw_face(image, boxes, names)                         
        cv2.imshow("result", image)
        key=cv2.waitKey(1)
        if key==27:
            break       
    #end of while.
    predictor.save_checkin_all("签到.txt")
    cv2.destroyAllWindows()
