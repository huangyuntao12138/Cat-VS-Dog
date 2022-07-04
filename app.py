import numpy as np
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def predict(img_path):
    # 载入模型
    model = load_model('./log/mod4.h5')
    # 载入图片，并处理
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100, 100))
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nor = img_RGB / 255
    img_nor = np.expand_dims(img_nor, axis=0)


    # 预测
    # print((np.argmax(model.predict(img_nor))))
    # print(model.predict(img_nor))
    y = model.predict(img_nor)
    img = Image.open(img_path)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure("猫狗识别系统")  # 图像窗口名称
    if y <= 0.5:
        plt.title('经识别，下图是：猫')  # 图像题目
    else:
        plt.title('经识别，下图是：狗')

    plt.imshow(img)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.show()


if __name__ == "__main__":
    predict('./Data1/test/2.jpg')
    predict('./Data1/test/100.jpg')
    predict('./Data1/test/221.jpg')
    predict('./Data1/test/222.jpg')
    predict('./Data1/test/520.jpg')
    predict('./Data1/test/88.jpg')
    predict('./Data1/test/188.jpg')
    predict('./Data1/test/1200.jpg')
    predict('./Data1/test/2222.jpg')
    predict('./Data1/test/1688.jpg')
