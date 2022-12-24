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
"""Train"""

import os
import argparse

from src.models import FaceNetModelwithLoss
from src.config import facenet_cfg
# from src.data_loader import get_dataloader
from src.data_loader_generate_triplets_online import get_dataloader
from src.eval_metrics import evaluate
# from src.eval_callback import EvalCallBack
# from src.LFWDataset import get_lfw_dataloader

import numpy as np

import mindspore.nn as nn
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
# from mindspore import  load_checkpoint, load_param_into_net
from mindspore import context
from mindspore.common import set_seed
set_seed(0)



parser = argparse.ArgumentParser(description='Face Recognition using Triplet Loss')

parser.add_argument("--data_url", type=str, default='D:/Desktop/Mindspore/facenet-master/datasets/train/')
parser.add_argument("--train_url", type=str, default="./result/")
parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=['Ascend', 'CPU'],
    help='device where the code will be implemented (default: CPU),若要在启智平台上使用NPU，需要在启智平台训练界面上加上运行参数device_target=Ascend')

# parser.add_argument("--data_triplets", type=str, default="D:/Desktop/Mindspore/facenet-master/datasets/train/vggface2.csv")
# parser.add_argument("--eval_root_dir", type=str, default="/data1/face/dataset/lfw_182/")
# parser.add_argument("--eval_pairs_path", type=str, default="/data1/face/dataset/LFW_pairs.txt")
# parser.add_argument("--eval_batch_size", type=int, default=64)
# parser.add_argument("--mode", type=str, default='train')
# parser.add_argument("--run_online", type=str, default='True')
# parser.add_argument("--is_distributed", type=str, default='False')
# parser.add_argument("--rank", type=int, default=0)
# parser.add_argument("--group_size", type=int, default=1)

args = parser.parse_args()



class InternalCallbackParam(dict):
    """Internal callback object's parameters."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def validate_lfw(model, lfw_dataloader):
    distances, labels = [], []

    # print("Validating on LFW! ...")
    for data in lfw_dataloader.create_dict_iterator():
        distance = model.evaluate(data['img1'], data['img2'])
        label = data['issame']
        distances.append(distance)
        labels.append(label)

    labels = np.array([sublabel.asnumpy() for label in labels for sublabel in label])
    distances = np.array([subdist.asnumpy() for distance in distances for subdist in distance])

    _, _, accuracy, _, _, _ = evaluate(distances=distances, labels=labels)
    # Print statistics and add to log
    print("Accuracy on LFW: {:.4f}+-{:.4f}\n".format(np.mean(accuracy), np.std(accuracy)))

    return accuracy


def main():
    cfg = facenet_cfg

    run_online = 'True'
    is_distributed = 'False'
    rank = 0
    group_size = 1


    # device_id = int(os.getenv('DEVICE_ID'))
    device_num = int(os.environ.get("RANK_SIZE", 1))
    if device_num == 1:
        is_distributed = 'False'
    elif device_num > 1:
        is_distributed = 'True'

    if is_distributed == 'True':
        print("parallel init", flush=True)
        init()
        rank = get_rank()
        args.group_size = get_group_size()
        context.reset_auto_parallel_context()
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=degree)
        context.set_auto_parallel_context(parameter_broadcast=True)

    if run_online == 'True':
        import moxing as mox
        local_data_url = '/cache/data/'
        mox.file.copy_parallel("obs://nanhang/shenao/data/dataset_eval/", local_data_url)
        local_triplets = local_data_url+"/vggface2.csv"
        local_train_url = "/cache/train_out/"
    else:
        local_data_url = args.data_url
        local_train_url = args.train_url
        local_triplets = args.data_triplets

    train_root_dir = local_data_url
    valid_root_dir = local_data_url
    train_triplets = local_triplets
    valid_triplets = local_triplets

    ckpt_path = local_train_url


    net = FaceNetModelwithLoss(num_classes=500, margin=cfg.margin, mode='train')
    # dict = load_checkpoint("/data1/face/FaceNet_mindspore/result/329_0/facenet-rank0-300_56.ckpt", net=net)
    # print("Loading the trained models from ckpt")
    # load_param_into_net(net, dict)

    # lr = nn.piecewise_constant_lr([50, 100, 150, 200],[0.0004, 0.00004, 0.000004, 0.0000004])

    optimizer = nn.Adam(net.trainable_params(), learning_rate=cfg.learning_rate)

    data_loaders, _ = get_dataloader(train_root_dir, valid_root_dir, train_triplets, valid_triplets,100,100,
                                     cfg.batch_size, cfg.num_workers, group_size, rank,
                                     shuffle=True, mode="train")
    data_loader = data_loaders['train']

    # lfw_dataloader = get_lfw_dataloader(eval_root_dir=args.eval_root_dir,
    #                                     eval_pairs_path=args.eval_pairs_path,
    #                                     eval_batch_size=args.eval_batch_size)

    loss_cb = LossMonitor(per_print_times=cfg.per_print_times)
    time_cb = TimeMonitor(data_size=cfg.per_print_times)

    # checkpoint save
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.per_print_times, keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(f"facenet-rank{rank}", ckpt_path + 'rank_' + str(rank), config_ck)

    callbacks = [loss_cb, time_cb, ckpoint_cb]

    # model = Model(net, optimizer=optimizer, amp_level="O2" )
    model = Model(net, optimizer=optimizer)

    # eval_callback = EvalCallBack(validate_lfw,
    #                              model,
    #                              lfw_dataloader,
    #                              interval=1
    #                              )
    # callbacks.append(eval_callback)

    print("============== Starting Training ==============")
    model.train(cfg.num_epochs, data_loader, callbacks=callbacks, dataset_sink_mode=True)

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    main()
