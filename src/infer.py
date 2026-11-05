import argparse
import functools
import os
import time

import cv2
import numpy as np
import torch
from PIL import ImageDraw, ImageFont, Image

from utils.face_detect import MTCNN
from utils.utils_train import add_arguments, print_arguments


class Predictor:

    def __init__(self, mtcnn_model_path, mobilefacenet_model_path, face_db_path, threshold=0.7):
        self.threshold = threshold
        self.mtcnn = MTCNN(model_path=mtcnn_model_path)
        self.device = torch.device("cuda") if torch.cuda.is_available()  else torch.device("cpu")
        # 加载模型
        self.model = torch.jit.load(mobilefacenet_model_path,map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        # 加载人脸库 & 计算特征
        self.features_db = self.load_face_db(face_db_path)

    def load_face_db(self, face_db_path):
        features = {}
        for path in os.listdir(face_db_path):
            name = os.path.basename(path).split('.')[0]
            image_path = os.path.join(face_db_path, path)
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            imgs, _ = self.mtcnn.infer_image(img)  # [ 【h,w,c】,... ]
            if imgs is None or len(imgs) > 1:
                print('人脸库中的 %s 图片包含不是1张人脸，自动跳过该图片' % image_path)
                continue
            imgs = self.process(imgs)              # [ 【h,w,c】 ]  => [ 【c,h,w】 ]     列表内部 array数组
            feature = self.infer(imgs[0])          #【1,512】         array数组
            features[name] = [ feature[0], 0]      #【512】   , 2024.12.1 JIA 签到修改， value修改为列表            
        return features

    @staticmethod
    def process(imgs):
        imgs1 = []
        for img in imgs:
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 127.5
            imgs1.append(img)
        return imgs1
    
    @staticmethod
    def cossim(x,y):
        s = np.dot(x,y)/np.sqrt(
              np.dot(x,x)*np.dot(y,y)
              )
        return s
    
    @staticmethod
    def add_text(img, text, x, y, color=(0, 0, 0), size=20):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('simfang.ttf', size)
        draw.text((x, y), text, color, font=font)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    
    # 预测图片
    def infer(self, imgs):
        assert len(imgs.shape) == 3 or len(imgs.shape) == 4
        if len(imgs.shape) == 3:
            imgs = imgs[np.newaxis, :]
        imgs = torch.tensor(imgs, dtype=torch.float32, device=self.device)
        features = self.model(imgs)
        features = features.detach().cpu().numpy()
        return features

        
    def recognition(self, img ):
        s = time.time()
        imgs, boxes = self.mtcnn.infer_image(img)
        # print('人脸检测时间：%dms' % int((time.time() - s) * 1000))
        if imgs is None:
            return None, None
        s = time.time()
        imgs = self.process(imgs)
        imgs = np.array(imgs, dtype='float32')
        features = self.infer(imgs)
        # print('特征计算时间：%dms' % int((time.time() - s) * 1000))
        s = time.time()
        names = []
        for f in features:
            results = []
            for name in self.features_db.keys():
                f_db, _ = self.features_db[name]
                prob    = self.cossim(f,f_db)
                results.append( [name,prob] )
            results = sorted(results, key=lambda d: d[1])
            result_max = results[-1] # 取最后一个（概率最大者），例如  ["刘德华"，0.7856]
            name,prob = result_max   # 分解 
            if prob > self.threshold:
                names.append(name) 
            else:
                names.append('unknow')
        # print('特征对比时间：%dms' % int((time.time() - s) * 1000))                
        return boxes, names

    #JIA 签到修改
    def get_checkin(self, name): 
        return self.features_db.get(name,["",-1])[1] #  签到  
    #JIA 签到修改            
    def set_checkin(self, name): 
        if name not in self.features_db.keys():   
            return 
        self.features_db[name][1]=1 #  签到  
    #JIA 签到修改 
    def save_checkin_all(self,filename): # {'刘德华': 0, '周星驰': 0, '李雪琴': 0, '杨幂': 0, '贾振堂': 1}
        with open(filename,"w",encoding='utf-8') as f:
            for name in self.features_db.keys():
                f.write("%s:%d\n"%(name,self.features_db[name][1]) )
           

    # 画出人脸框和关键点
    def draw_face(self, image, boxes, names ):
        if boxes is not None:
            for i in range(boxes.shape[0]):
                bbox = boxes[i, :4]
                name = names[i]
                x1,y1, x2, y2 = bbox.astype(int)    
                color = (0, 255, 0) if self.features_db.get(name,["",0])[1] else (255,0,0)   #JIA 签到修改
                # 画人脸框
                cv2.rectangle( image, (x1,y1),(x2,y2), color, 1)
                # 人的名字
                image = self.add_text(image, name, x1, y1-40, color=color, size=30)
        else:
            image = self.add_text(image, "???",10,50, color=(255, 255,0), size=30)
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('image_path',               str,     'images_test/合照.bmp',      '预测图片路径')
    add_arg('face_db_path',             str,     'images_db',                 '人脸库路径')
    add_arg('threshold',                float,   0.5,                         '判断相识度的阈值')
    add_arg('mobilefacenet_model_path', str,     'save_model/mobilefacenet.pth',  'MobileFaceNet预测模型的路径')
    add_arg('mtcnn_model_path',         str,     'save_model/mtcnn',          'MTCNN预测模型的路径')
    args = parser.parse_args()
    print_arguments(args)
    
    #########
    
    predictor = Predictor(args.mtcnn_model_path, args.mobilefacenet_model_path, args.face_db_path, threshold=args.threshold)
    start = time.time()
    image = cv2.imdecode(np.fromfile(args.image_path, dtype=np.uint8), -1)
    boxes, names = predictor.recognition(image)
    if boxes is not None:
        #print('预测的人脸位置：', boxes.astype(np.int_).tolist())
        print('识别的人脸名称：', names)
        print('总识别时间：%dms' % int((time.time() - start) * 1000))
    image = predictor.draw_face(image, boxes, names)
    cv2.imshow("result", image)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()