import torch.nn as nn
import torch
import numpy as np

#coding=gbk
class Bottleneck(nn.Module):  # �����50��101��152��resnet��ÿһ���в�ṹ(��������+һ���в�)
    expansion = 4  # ͬһ���в�ṹ�еĵ�����ľ���˵ĸ������ڵ�1,2��ľ���˵ĸ�����4��

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm1d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False,
                               padding=1)  # ����ͬ���������Ϊ�в��ĵ�һ���в�ṹ��stride=2���Ӷ���input_size��СΪԭ����һ��
        self.bn2 = nn.BatchNorm1d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel * self.expansion,  # ��Ϊ���������������������4��
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm1d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=5, include_top=True):
        '''
        ��������
        block��ResNet��ʹ�õĲв�ṹ���ͣ����resnet��18,34��ľ�ʹ��BasicBlock,50,101,152ʹ��Bottleneck����
        blocks_num��ÿһ���в��(��������в�ṹ)��߰����Ĳв�ṹ�ĸ�����Ȼ�������������Щ�������б�
        num_classes: �������
        include_top: ����Ϊ����resnet�����ϴ���Ӹ��ӵ�����Ļ���
        '''
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # ��������ͼ�����(������ʼ��maxpooling֮�������ͼ)

        self.conv1 = nn.Conv1d(1, self.in_channel, kernel_size=7, stride=2,  # ע��Ŷ��self.in_channel����Ϊ��һ���������������
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)  # ���ػ�����
        self.layer1 = self._make_layer(block, 64, blocks_num[0])  # ��һ���в�� Conv_2
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)  # �ڶ����в�� Conv_3
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  # �������в�� Conv_4
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # ���ĸ��в�� Conv_5
        if self.include_top:
            # self.avgpool = nn.AdaptiveAvgPool1d((1, 1))  # output size = (1, 1)
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):  # stride=1ָ����strideĬ����1����˼
        '''
        ��������
        block: �в�ṹ������(basic bottle)
        channel: �в�ṹ�о�������õ�����˵ĸ���(������ǵ�һ�����˵ĸ���)
        block_num�� ��ǰ�в��Ĳв�ṹ�ĸ���
        '''

        downsample = None  # �²���������ʼ��
        '''
        ��ΪConv_2,Conv_3...����ͨ��_make_layer���������ɵģ�������ÿһ�㶼��Ҫ���²��������ģ�Ҳ����ÿһ���ÿһ���в�ṹ����Ҫ�²����ĺ�����Ҳ���Ǵ����߲в�ṹ������������¼���:
        1. �����ResNet�е�50,101,152,��ôÿһ���в��ĵ�һ���Ĳв�ṹ��һ���Ǵ����ߵĲв�ṹ
        2. �����ResNet�е�34����ôֻ�д�Conv_3��ʼ��ÿһ���в��ĵ�һ���в�ṹ���Ǵ����ߵĲв�ṹ

        �����ù���ResNet34��ResNet50�Ĺ�����̾���˵��
        1. ResNet34
        (1)Conv_2����Ϊ��ʱstride=1,�Ҵ�ʱself.in_channel�ʹ����������channel����64�����Բ��������²���������Ȼ��ͨ��ifģ��������Ǽ��д��빹���˵�һ���в�ṹ����ʱ�����stride=1,
        ������ѭ������������Conv_2�е������Ĳв�ṹ��ע�⣬��ʱ�����strideҲ��1��˵��û�иı�width,height
        (2)Conv_3: ��ʱ��self.in_channel����64�������������channel��128�ˣ�������Ϊ�����stride�����2���������������������Թ������²����������²��������ľ����������channel*1��
        ���Ի���128��Ϊ��ʱ�����stride��2�������²���������Ҳ��2���𵽽�ԭ�����width��height����һ������ã�����ڹ������ߵĵ�һ���в�ṹ��ʱ�򣬵�һ��������stride���õ�2����
        ��Сshape�����ã��ڶ���������ֱ��stride=1,ά��ԭ������Ҫע�⹹�����һ���в�ṹ֮�󣬽�self.in_channel���£�����self.in_channel�����128��Ȼ���ڹ���ʣ��Ĳв�ṹ��ʱ��
        �������������128��strideĬ����1����һֱ����ԭ����
        (3)Conv_4,Conv_5ͬ��

        2. ResNet50
        (1)Conv_2: �����stride=1�����Ǵ�ʱself.in_channel=64,channel=64,block.expansion=4,���Թ������²���������������ΪConv_2��û�иı�input��width��height�������²���������strideҲ��1
        �ڹ����һ���в�ṹʱ����һ��������stride�ǹ̶���1�ģ�Ȼ��ڶ����������Ϊ�����stride����1���Ի���1�����������stride���ǹ̶�1,�������һ���в�ṹ֮���޸�self.in_channel=256��
        �ڹ���Conv_2�������в�ṹʱ������Ĭ��stride����1����ÿһ���в�ṹ�ľ���˵������������(256,64),(64,64),(64,256)
        (2)Conv_3: �����stride=2����self.in_channel=256,channel=128,block.expansion=4,���Թ���������256�������512��stride=2�ľ������Ϊ�²�����Ȼ���ڹ�����һ���в�ṹʱ�����ߵĵ�һ�������
        ��������256�������128��ͬʱ��Ϊstride=1����shapeû�䣬�ڶ�������������128�������128��������Ϊ��ʱstride���Ǵ����2������shape��Ϊԭ����һ�룬�����㣬��Ϊstride=1��shape���䣬����channel
        Ϊ128�����Ϊ128*4=512�������͹�����˵�һ���в�ṹ����һ��������в�ṹ����ֱ�ӱ���shape���䣬�������channels��������(512,128),(128,128),(128,512)��
        (3)Conv_4,Conv_5��Conv_3ͬ��
        '''

        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(  # �����²�������
                nn.Conv1d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                # ���߲в����˸���Ҫ��4��
                nn.BatchNorm1d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion  # Ҫ��ʱ������Ϊ��һ���в�ṹ������ֵ

        for _ in range(1, block_num):  # ����ʵ�ֵĲв�ṹ
            layers.append(block(self.in_channel, channel))

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

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet50(num_classes=5, include_top=True):
    return ResNet(Bottleneck, [2, 3, 5, 2], num_classes=num_classes, include_top=include_top)