# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F


class RV1Torch(nn.Module):

    def __init__(self, output=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.dense1 = torch.nn.Linear(64, 256)  #64 Conv with (3*3) followed by a (3*3) max pooling
        self.dense2 = torch.nn.Linear(256, 256)
        self.dense3 = torch.nn.Linear(256, output)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 3)
        # add dropout (λ = 0.3)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 3)
        # add dropout (λ = 0.3)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = RV1Torch()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

context = 32
n_bins = 32
input = torch.randn(1, 1, context, n_bins)
out = net(input)
print(out)

target = torch.tensor([0,1,1,0,0], dtype=torch.long)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output

loss = nn.CrossEntropyLoss()(out, target)



class RV1TF(object):
    """
    Richard Vogl model
    http://ifs.tuwien.ac.at/~vogl/
    """

    # def __init__(self):
    #     # self.model = self.createModel()
    #     pass

    def createModel(self, context=25, n_bins=256, output=5):
        """Return a ts model based 
        
        Keyword Arguments:
            context {int} -- [description] (default: {25})
            n_bins {int} -- [description] (default: {84})
            output {int} -- number of classes in the output (should be the events: 36, 40, 41, 46, 49) (default: {5})
            outputWeight {list} --  (default: {[]}) 
        
        Returns:
            [type] -- [description]
        """
        # When to apply the dropout?
        # How to handle the bidirectional aggregation ? Sum, or nothing ?
        # How to handle the context for the learning 400 samples before learning?

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), input_shape=(context, n_bins, 1), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
            tf.keras.layers.Dropout(0.3),
            # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
            # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
            # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(output, activation=tf.keras.activations.sigmoid)
        ])

        model.compile(
            optimizer="adam",  #tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss=tf.keras.backend.binary_crossentropy,  #tf.compat.v2.nn.sigmoid_cross_entropy_with_logits, those two are equivalent if binary_crossentropy(logits=True), since the activation is already the probabilty because of the sigmoid and not logits, do not use those
            metrics=["accuracy", tf.keras.metrics.BinaryAccuracy()]  # PeakPicking()
        )
        return model


# rv1 = RV1()

# """
# load
# """
# import pickle

# file="/home/mickael/Documents/programming/madmom-0.16.dev0/madmom/models/drums/2018/drums_cnn0_O8_S0.pkl"
# with open(file, "rb") as f:
#     u = pickle._Unpickler(f)
#     u.encoding = 'latin1'
#     p = u.load()
#     print(p)

# 00:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x7fd5b2f38ac8>
# 01:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd5722e5d30>
# 02:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x7fd5722e5cf8>
# 03:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd5722e5f28>
# 04:<madmom.ml.nn.layers.MaxPoolLayer object at 0x7fd5b3eca978>
# 05:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x7fd5b3eca6d8>
# 06:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd5b3eca438>
# 07:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x7fd5b3eca048>
# 08:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd5b3ec3da0>
# 09:<madmom.ml.nn.layers.MaxPoolLayer object at 0x7fd57228d208>
# 10:<madmom.ml.nn.layers.StrideLayer object at 0x7fd57228d278>
# 11:<madmom.ml.nn.layers.FeedForwardLayer object at 0x7fd57228d2b0>
# 12:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd57228d390>
# 13:<madmom.ml.nn.layers.FeedForwardLayer object at 0x7fd57228d588>
# 14:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd57228d6a0>
# 15:<madmom.ml.nn.layers.FeedForwardLayer object at 0x7fd57228d860>
