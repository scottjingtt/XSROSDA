import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

# 1. Backbone
vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, "vgg16":models.vgg16, "vgg19":models.vgg19,
"vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn, "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn}
class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.in_features = model_vgg.classifier[6].in_features

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50,
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        featmap = self.layer4(x)
        x = self.avgpool(featmap)
        x = x.view(x.size(0), -1)
        return x, featmap


class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()

        model_resnet = models.resnet50(pretrained=True)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y

# 2. BaseModel

class BaseFeature(nn.Module):
    def __init__(self, output_dim, input_dim, hidden_dim=256):
        super(BaseFeature, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Dropout(), nn.ReLU(inplace=True))
        self.fc1.apply(init_weights)
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, output_dim))
        self.fc2.apply(init_weights)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Classifier(nn.Module):
    def __init__(self, class_num, input_dim, hidden_dim=256):
        super(Classifier, self).__init__()
        self.type = type
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Dropout(), nn.ReLU(inplace=True))
        self.fc1.apply(init_weights)

        self.fc2 = nn.Linear(hidden_dim, class_num)
        self.fc2.apply(init_weights)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x



class AttributeProjector(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat):
        super(AttributeProjector, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(in_features=in_feat, out_features=hidden_feat), nn.ReLU())
        self.fc1.apply(init_weights)
        self.fc2 = nn.Linear(in_features=hidden_feat, out_features=out_feat)
        self.fc2.apply(init_weights)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        logit = self.fc2(x)
        x = self.sigmoid(logit)

        return x, logit

# 3. ProtoModel
class PPNet(nn.Module):
    def __init__(self, img_size, prototype_shape,
                 num_classes, init_weights=True,
                 dist_type = 'cosine',
                 prototype_activation_function=None,
                 add_on_layers_type='regular',
                 take_best_prototypes=True, t=0.8, args=None):
        '''

        Args:
            features:
            img_size: 7 (feature size: 7 x 7)
            prototype_shape: (mk*C) x 512 x 1 x 1
            num_classes: 85/337 for AwA2/LAD, dimension of attributes
            init_weights: True
            dist_type: euclidean | cosine (euc is oroginal protoNet design; for multi-label, cosine might work)
            prototype_activation_function: ['log', 'linear', other methods convert distances to similarities]
            add_on_layers_type: 'regular', 1 layer 1x1 conv layer
            take_best_prototypes: True
            t: variance to hardtanh function
            args: opts
        '''

        super(PPNet, self).__init__()
        self.take_best_prototypes = take_best_prototypes # true (?)
        self.img_size = img_size # featuremap, 7
        self.prototype_shape = prototype_shape # 85*3 x 512 x 1 x 1
        self.num_prototypes = prototype_shape[0] # 85 * 3
        self.num_classes = num_classes # here the num_class means attributes classes
        self.epsilon = 1e-4
        self.thresholds = torch.zeros((1, prototype_shape[0]))
        self.args = args
        self.prototype_activation_function = prototype_activation_function # 'log'
        self.dist_type = dist_type # [cosine | euclidean]

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class/element of attributes
        e.g., 3 prototypes for each attribute, not class. 3 * 85 in total
        '''
        assert (self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes).cuda()

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        assert num_prototypes_per_class == self.args.prototypes_K
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.t = t
        self.binarizer = torch.nn.Hardtanh(min_val=self.t, max_val=1 + self.t)

        # resnet output is 2048; vgg is 4096
        first_add_on_layer_in_channels = 2048

        if add_on_layers_type == 'bottleneck':
            raise Exception("Error add-on layer!")
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert (current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1],
                          kernel_size=1),
                nn.ReLU(),
                # could use only one layer
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                # nn.ReLU()
                # nn.Sigmoid()
            )

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        # attributes, self.num_classes = 85
        # for positive attributes
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False)  # do not use bias
        # for negative/0 attributes
        self.last_layer_negative = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False)

        if init_weights:
            self._initialize_weights() # set correct connections as 1s, incorrect connections as 0s
        print(self.last_layer)
        print(self.last_layer_negative)

    def forward(self, x):
        distances, conv_features = self.prototype_distances(x) # bs x w x h x prot_num
        # global min pooling
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes) # bs x prot_num
        prototype_activations = self.distance_2_similarity(min_distances)

        # if residual - add residual layer as ProtoProp did

        att_logits = self.last_layer(prototype_activations) # positive attributes logit
        att_logits_neg = self.last_layer_negative(prototype_activations)
        # Use attributes to seen classes labels linear mapping, directly get the prediction for classes
        cls_logits = torch.cat([att_logits_neg.unsqueeze(2), att_logits.unsqueeze(2)], dim=2) # n x att_dim x 2
        # Because negative means 0, positive means 1. so put neg at first
        return att_logits, cls_logits, min_distances, conv_features

    def push_forward(self, x):
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
        return conv_features, distances


    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        if self.dist_type == 'euclidean':
            distances = self._l2_convolution(conv_features)
        elif self.dist_type == 'cosine': # here
            x = conv_features / conv_features.norm(dim=1, keepdim=True)
            weight = self.prototype_vectors / self.prototype_vectors.norm(dim=1, keepdim=True)
            sim = F.conv2d(x, weight=weight) #self.logit_scale.exp() *
            distances = 1 - sim
            # This is actually cosine similarity [-1, 1]
        elif self.dist_type == 'dot': # dot is already similarity actually
            sim = -F.conv2d(conv_features, weight=self.prototype_vectors)
            distances = -sim
        else:
            raise Exception("ProtoNet dist_type is not correct!")
        return distances, conv_features


    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def conv_features(self, featmap):
        '''
        the feature input to prototype layer
        '''
        # x = self.features(x) # backbone features
        x = self.add_on_layers(featmap)
        return x


    def distance_2_similarity(self, distances):
        if self.dist_type == 'euclidean':
            if self.prototype_activation_function == 'log':
                return torch.log((distances + 1) / (distances + self.epsilon))
            elif self.prototype_activation_function == 'linear':
                return -distances
            else:
                return self.prototype_activation_function(distances)
        elif self.dist_type == 'cosine':
            # or simply use sim = 1 - distances
            return 1 - distances # (-1,1) #(1 - distances) / 2 # change (-1,1) -> (0, 2) --> (0, 1)
        else:
            raise Exception("The distances to similarity function is not correct!")

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

        if self.last_layer_negative:
            self.last_layer_negative.weight.data.copy_(
                correct_class_connection * positive_one_weights_locations*0
                + incorrect_class_connection * negative_one_weights_locations*0)
