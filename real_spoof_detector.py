import os
import glob
import time
import cv2
import numpy as np
import dlib
from imutils import face_utils
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torchvision
import math
from PIL import Image
import tkinter as tk
from tkinter import filedialog

BN = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BN(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# AENet_C,S,G is based on ResNet-18
class AENet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1000, sync_stats=False):

        global BN

        self.inplanes = 64
        super(AENet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BN(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # Three classifiers of semantic informantion
        self.fc_live_attribute = nn.Linear(512 * block.expansion, 40)
        self.fc_attack = nn.Linear(512 * block.expansion, 11)
        self.fc_light = nn.Linear(512 * block.expansion, 5)
        # One classifier of Live/Spoof information
        self.fc_live = nn.Linear(512 * block.expansion, 2)

        # Two embedding modules of geometric information
        self.upsample14 = nn.Upsample((14, 14), mode='bilinear')
        self.depth_final = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.reflect_final = nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=1, bias=False)
        # The ground truth of depth map and reflection map has been normalized[torchvision.transforms.ToTensor()]
        self.sigmoid = nn.Sigmoid()

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BN(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        depth_map = self.depth_final(x)
        reflect_map = self.reflect_final(x)

        depth_map = self.sigmoid(depth_map)
        depth_map = self.upsample14(depth_map)

        reflect_map = self.sigmoid(reflect_map)
        reflect_map = self.upsample14(reflect_map)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x_live_attribute = self.fc_live_attribute(x)
        x_attack = self.fc_attack(x)
        x_light = self.fc_light(x)
        x_live = self.fc_live(x)

        return x_live

class CelebASpoofDetector(ABC):
    def __init__(self):
        """
        Participants should define their own initialization process.
        During this process you can set up your network. The time cost for this step will
        not be counted in runtime evaluation
        """

    @abstractmethod
    def predict(self, image):
        """
        Process a list of image, the evaluation toolkit will measure the runtime of every call to this method.
        The time cost will include any thing that's between the image input to the final prediction score.
        The image will be given as a numpy array in the shape of (H, W, C) with dtype np.uint8.
        The color mode of the image will be **RGB**.

        params:
            - image (np.array): numpy array of required image
        return:
            - probablity (float)
        """
        pass

def pretrain(model, state_dict):
    own_state = model.state_dict()

    for name, param in state_dict.items():
        realname = name.replace('module.','')
        if realname in own_state:
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[realname].copy_(param)
            except:
                print('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(realname, own_state[name].size(), param.size()))
                print("But don't worry about it. Continue pretraining.")

class TSNPredictor(CelebASpoofDetector):
    def __init__(self):
        self.num_class = 2
        self.net = AENet(num_classes=self.num_class)
        checkpoint = torch.load('./ckpt_iter.pth.tar', map_location='cpu')  # ./ckpt_iter.pth.tar

        pretrain(self.net, checkpoint['state_dict'])

        self.new_width = self.new_height = 224

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.new_width, self.new_height)),
            torchvision.transforms.ToTensor(),
        ])

        self.net.cpu()  # net.cuda(). Phiên bản torch ko hỗ trợ cuda
        self.net.eval()

    def preprocess_data(self, image):
        processed_data = Image.fromarray(image)
        processed_data = self.transform(processed_data)
        return processed_data

    def eval_image(self, image):
        data = torch.stack(image, dim=0)
        channel = 3
        input_var = data.view(-1, channel, data.size(2), data.size(3)).cpu()  # cuda()
        with torch.no_grad():
            rst = self.net(input_var).detach()
        return rst.reshape(-1, self.num_class)

    def predict(self, images):
        real_data = []
        for image in images:
            data = self.preprocess_data(image)
            real_data.append(data)
        rst = self.eval_image(real_data)
        rst = torch.nn.functional.softmax(rst, dim=1).cpu().numpy().copy()
        probability = np.array(rst)
        return probability

def crop_boundary(top, bottom, left, right, faces):
    if faces:
        top = max(0, top - 200)
        left = max(0, left - 100)
        right += 100
        bottom += 100
    else:
        top = max(0, top - 50)
        left = max(0, left - 50)
        right += 50
        bottom += 50
    return (top, bottom, left, right)

def crop_face(imgpath):
    frame = cv2.imread(imgpath)
    if frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detect = dlib.get_frontal_face_detector()
    rects = face_detect(gray, 1)
    if not len(rects):
        return None

    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        top, bottom, left, right = crop_boundary(y, y + h, x, x + w, len(rects) <= 2)
        crop_img = frame[top:bottom, left:right]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
    return crop_img

# def open_file_dialog():
#     root = tk.Tk()
#     root.withdraw()
#
#     file_path = filedialog.askopenfilename()
#     if file_path:
#         print("Selected file:", file_path)
#     return file_path

def generator(n1, n2):
    yield n1, n2

if __name__ == "__main__":
    before = time.time()
    # ls = open_file_dialog()
    image_path = "image.jpg"  ######## INPUT IMAGE DIRECTORY HERE
    img = crop_face(image_path)
    ls = [img]

    final_image = []
    final_image_id = []

    for image_id in ls:
        image = image_id
        final_image.append(image)
        final_image_id.append(image_path)

    np_final_image_id = np.array(final_image_id)
    np_final_image = np.array(final_image, dtype="uint8")  # "object" or "uint8"

    detector = TSNPredictor()
    output_probs = {}
    image_iter = generator(np_final_image_id, np_final_image)

    for image_id, image in image_iter:
        prob = detector.predict(image)
        for idx, i in enumerate(image_id):
            output_probs[i] = float(prob[idx][1])
    after = time.time()
    print(output_probs)
    print("Elapsed time: {} seconds".format((after - before) / len(ls)))
    print(len(ls), 'images')
