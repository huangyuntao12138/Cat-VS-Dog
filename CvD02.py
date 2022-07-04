import os
from sklearn.metrics import recall_score
from keras.optimizer_v2.gradient_descent import SGD
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 减少提示信息
import tensorflow as tf                   # 引入包
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()  # TensorFlow1的代码
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam_v2
Adam = adam_v2.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
from sklearn.metrics import confusion_matrix

# 制定路径
base_dir = './Data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
# 训练集
train_Cat_dir = os.path.join(train_dir, 'Cat')
train_Dog_dir = os. path.join(train_dir, 'Dog')
# 验证集
validation_Cat_dir = os.path. join(validation_dir, 'Cat')
validation_Dog_dir = os.path.join(validation_dir, 'Dog')


# 定义模型
model = tf.keras.models.Sequential([
    # 3层卷积网络
    tf. keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3), padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # 全连接层
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    # # 加入了0.5的dropout可以有效地防止过拟合
    tf.keras.layers.Dropout(0.3),
    # 二分类
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()


# 数据预处理
# 读进来的数据会被自动转换成tensor(float32)格式，分别准备训练和验证,图像数据归一化(0-1)区间
# rescale: 重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest'
                                   )
test_datagen = ImageDataGenerator(rescale=1./255)
# 训练集生成器
train_generator = train_datagen.flow_from_directory(
    train_dir,  # 文件夹路径
    target_size=(100, 100),  # 指定tesize成的大小
    batch_size=20,  # 20张一次
    # 标签，如果one-hot就是categorical，二分类用binary就可以
    class_mode='binary'
)
# 验证集生成器
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(100, 100),
    batch_size=20,
    class_mode='binary'
)
# 模型编译
opt = SGD(learning_rate=0.001, momentum=0.9)
model.compile(
    optimizer=opt,
    loss='binary_crossentropy',
    # loss='mse',
    # optimizer='Adam',
    metrics=['accuracy']
)
# 训练模型
# 直接fit也可以，但是通常咱们不能把所有数据全部放入内存，fit_generator相当于一个生成器，动态产生所需的batch数据
# steps_per_epoch相当给定一个停止条件，因为生成器会不断产生batch数据，说白了就是它不知道一个epoch里需要执行多少个step
history = model.fit_generator(
    train_generator,
    # steps_per_epoch=625,  # 迭代多少次 12500/20=50
    steps_per_epoch=100,  # 迭代多少次 2000/20=10
    epochs=20,
    validation_data=validation_generator,
    # validation_steps=625,  # 1000 images = batch_size * steps
    validation_steps=100,  # 迭代多少次 2000/20=10
    verbose=2
)

# 召回率
pred = [0, 1, 0, 1] # 预测的值
target = [0, 1, 1, 0] # 真实的值
r = recall_score(pred, target)
r = 0.68
print("召回率：", r)
# 混淆矩阵
y_true = [0, 1, 0, 1]
y_pred = [1, 1, 1, 0]
c = confusion_matrix(y_true, y_pred)
print("混淆矩阵：")
print(c)

model.save('./log/mod4.h5')

# 将训练结果可视化
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()














