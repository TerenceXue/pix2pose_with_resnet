# 西安交通大学 智造未来2201 薛天赐
# 开发时间 2024/2/28 23:08
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow.keras import layers, optimizers, Sequential, Model
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt

#配置显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存 
session = tf.compat.v1.Session(config=config)
print(device_lib.list_local_devices())

class DataLoader:
    def __init__(self, directory):
        self.directory = directory

    def load_data(self):
        data = []
        filenames = os.listdir(self.directory)
        for filename in filenames:
            if filename.endswith(".npz"):
                file_path = os.path.join(self.directory, filename)
                loaded_data = np.load(file_path)
                data.append(loaded_data)
        return data


directory = r"/data15/xuetc2401/domrand/CNN/data_bag/test_push"
data_loader = DataLoader(directory)
loaded_data = data_loader.load_data()
all_depth_array=[]#所有文件的深度图像，一张一张的
all_rgb_array=[]#所有文件的rgb图像，一张一张的
all_pos=[]#所有文件的posA图像，一个一个的
for i, data in enumerate(loaded_data):
    obs_depth_array = data['Depth']
    obs_rgb_array = data['Rgb']
    pos = data['Pos']
    print(f"Data from file {i+1}:")
    for i in pos:
        for env_pos in i:
            all_pos.append(env_pos)
    for depth_data in obs_depth_array:
        all_depth_array.append(depth_data)
    for rgb_data in obs_rgb_array:
        all_rgb_array.append(rgb_data)
all_depth_array=np.array(all_depth_array)
all_rgb_array=np.array(all_rgb_array)
all_pos_array=np.array(all_pos)

# data preprocess

X = all_depth_array
num = len(X)
X_list = []
for i in range(num):
    X_list.append((X[i]-np.min(X[i])) / (np.max(X[i])-np.min(X[i])))  # 把深度数据线性标准化到[0,1]之间
X = np.array(X_list).reshape(800,240,360,1)
Y=all_pos_array

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_db = train_db.shuffle(1000).batch(16)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.batch(16)

sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))
print(tf.__version__)
# 定义网络结构
class BasicBlock(layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):

        # [b, h, w, c]
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        identity = self.downsample(inputs)

        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


class ResNet(Model):

    def __init__(self, layer_dims, output=3):  # [2, 2, 2, 2]
        super(ResNet, self).__init__()

        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])

        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # output: [b, 512, h, w],
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(output)

    def call(self, inputs, training=None):
        x = self.stem(inputs, training=training)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        # [b, c]
        x = self.avgpool(x)
        # [b, 100]
        x = self.fc(x)

        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        # may down sample
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks


def resnet18():  # 两种resnet结构
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 3])


tf.random.set_seed(2345)



# [b, 32, 32, 3] => [b, 1, 1, 512]
model = resnet18()
model.build(input_shape=(None, 240, 360, 1))#根据输入图片的尺寸更改
model.summary()
optimizer = optimizers.Adam(lr=1e-3)
all_acc=[]
max_acc=[]
for epoch in range(10):

    for step, (x, y) in enumerate(train_db):

        with tf.GradientTape() as tape:
            # [b, 32, 32, 3] => [b, 6]
            logits = model(x, training=True)
            # [b] => [b, 6]

            # compute loss
            loss = tf.losses.mean_squared_error(y, logits)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 4 == 0:
            print('epoch',epoch,'step', step, 'loss:', float(loss))

    for x, y in test_db:

        logits = model(x, training=False)

        pred = logits

        total_correct = []
        for j in range(len(y)):  # testsize
            correct = []
            for i in range(len(y[j])):
                correct.append(1-abs(float(pred[j][i]) - float(y[j][i]) / float(y[j][i]))) # 三个位置的准确率相加
            mean_correct = np.mean(correct)  # 三个位置的准确率平均
            total_correct.append(mean_correct)
        mean_acc = np.mean(total_correct)  # 测试集的size
        print('epoch:', epoch, 'acc:', mean_acc)
        all_acc.append(mean_acc)
        if mean_acc==max(all_acc):
            model.save_weights(f'/data15/xuetc2401/domrand/CNN/saved_weights/weights_{mean_acc}.ckpt')
            print('saved')
            max_acc.append(mean_acc)
            break
plt.plot(range(10),max_acc)
plt.show()