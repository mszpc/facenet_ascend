# 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""FaceNet"""
from src.resnet import resnet50
from src.loss import TripletLoss, PairwiseDistance

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore import dtype as mstype


class FaceNetModel(nn.Cell):
    """
    the densenet121 architecture
    """

    def __init__(self, num_classes, margin, mode):
        super(FaceNetModel, self).__init__()
        self.margin = margin
        self.mode = mode
        self.resnet50_backbone = resnet50(class_num=num_classes)
        embedding_size = 128
        # num_classes = 500
        self.l2_dist = PairwiseDistance(2)
        self.resnet50_backbone_fc = nn.SequentialCell(
            nn.Flatten(),
            nn.Dense(100352, embedding_size))

        self.resnet50_backbone_classifier = nn.Dense(embedding_size, num_classes)
        self.flatten = nn.Flatten()
        self.gather = P.Gather()
        self.squeeze = P.Squeeze()
        self.tuple_to_tensor = P.TupleToArray()

    def l2_norm(self, ft):
        input_size = ops.shape(ft)
        buffer = ops.pows(ft, 2)
        normp = ops.reduce_sum(buffer, 1)
        normp = normp + 1e-10
        norm = ops.sqrt(normp)
        norm = norm.view(-1, 1).expand_as(ft)
        _output = ops.div(ft, norm)
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = self.resnet50_backbone(x)
        x = self.resnet50_backbone_fc(x)

        features = self.l2_norm(x)
        alpha = 10
        features = features * alpha
        return features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.resnet50_backbone_classifier(features)
        return res

    def construct(self, anc_img, pos_img, neg_img, pos_class, neg_class):
        anc_embed = self.forward(anc_img)  # (8, 128)
        pos_embed = self.forward(pos_img)  # (8, 128)
        neg_embed = self.forward(neg_img)  # (8, 128)
        pos_dist = self.l2_dist(anc_embed, pos_embed)
        neg_dist = self.l2_dist(anc_embed, neg_embed)

        all_index = self.flatten(P.Less()((neg_dist - pos_dist), self.margin))
        all_index = P.Cast()(all_index, P.DType()(anc_embed))
        anc_hard_embed = anc_embed
        pos_hard_embed = pos_embed
        neg_hard_embed = neg_embed
        return anc_hard_embed, pos_hard_embed, neg_hard_embed, all_index

    def evaluate(self, data_a, data_b):
        embed_a = self.forward(data_a)
        embed_b = self.forward(data_b)
        distance = self.l2_dist(embed_a, embed_b)
        return distance


class FaceNetModelwithLoss(nn.Cell):
    def __init__(self, num_classes, margin, mode):
        super(FaceNetModelwithLoss, self).__init__()
        self.network = FaceNetModel(num_classes=num_classes, margin=margin, mode=mode)
        self.loss = TripletLoss(margin)
        self.cast = P.Cast()
        print("Model has been built")

    def construct(self, anc_img, pos_img, neg_img, pos_class, neg_class):
        anc_hard_embed, pos_hard_embed, \
        neg_hard_embed, all_index = self.network(anc_img, pos_img, neg_img, pos_class, neg_class)
        total_loss = self.loss(anc_hard_embed, pos_hard_embed, neg_hard_embed, all_index)
        return self.cast(total_loss, mstype.float32)

    def evaluate(self, img1, img2):
        dist = self.network.evaluate(img1, img2)
        return dist
