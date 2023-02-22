import cv2
import numpy as np


classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


class detection():
    def __init__(self, param_path, weight_path, height, width, num_classes):
        self.inpWidth = height
        self.inpHeight = width
        self.net = cv2.dnn.readNet(param_path, weight_path)
        self.mean = np.array([104, 117, 123], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([1, 1, 1], dtype=np.float32).reshape(1, 1, 3)
        self.keep_ratio = True

    def resize_image(self, srcimg):
        padh, padw, newh, neww = 0, 0, self.inpHeight, self.inpWidth
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padw = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, padw, self.inpWidth - neww - padw, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padh = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, padh, self.inpHeight - newh - padh, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, padh, padw

    def _normalize(self, img):  ### c++: https://blog.csdn.net/wuqingshan2010/article/details/107727909
        # img = img.astype(np.float32) / 255.0
        img = (img - self.mean) * self.std
        return img

    def detect(self, srcimg):
        img, newh, neww, padh, padw = self.resize_image(srcimg)
        img = self._normalize(img)
        blob = cv2.dnn.blobFromImage(img)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # outs[]->[][classes][confidence][x][y][x][y]
        # inference output
        outimg = srcimg.copy()
        height, width, _ = outimg.shape
        for out in outs[0][0]:
            for ou in out:
                _, cl, conf, top, left, bottom, right = ou
                if cl < 1:
                    continue
                cv2.rectangle(outimg, (int(height*top), int(width*left)), (int(height*bottom), int(width*right)), (0, 0, 255), 3)
                cv2.putText(outimg, classes[int(cl)-1]+':'+str(conf), (int(height*top), int(width*left)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)

        return outimg


def predict(param_path, weight_path, imgpath, height, width):

    ssdnet = detection(param_path, weight_path, height, width, 2)
    srcimg = cv2.imread(imgpath)
    outimg = ssdnet.detect(srcimg)
    cv2.imshow('origin', srcimg)
    cv2.imshow('segment', outimg)
    cv2.waitKey()
    # cv2.imwrite(imgpath.replace('.jpg', 'result.jpg'), outimg)


if __name__ == "__main__":
    height = 512
    width = 512
    root = '/media/z590/G/detection/ssd-pytorch-master/models/caffemodel/'
    param_path = root + 'SSDJacintoNetV2.prototxt'
    weight_path = root + 'SSDJacintoNetV2.caffemodel'

    imgpath = "/media/z590/G/detection/ssd-pytorch-master/img/street.jpg"
    predict(param_path, weight_path, imgpath, height, width)




