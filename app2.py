from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array,load_img

# 种类字典
label = np.array(['cat', 'dog'])
# 载入模型
model = load_model('./log/mod2.h5')
# 导入图片
image = load_img('./Data1/test/1.jpg')

