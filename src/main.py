#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging


from User.serverrep import FedRep
from User.serveravg import FedAvg
from User.serverserdef import FedSD
from User.serverproto import FedProto
from User.serverpd import FedPD
from User.servermk import FedMK
from User.servertrimmed import FedTrimmed
from User.serverRepTrim import FedRepTrim
from User.serverRepTrimSD import FedRepTrimSD
from User.serverflip import FedFLIP
from User.servermedian import FedMedian
from User.serverbulyan import FedBulyan
from User.serverapple import APPLE
from User.serverCLCM import FedCLCM




from model.models import *
from model.resnet import *
from model.Resnet import ResNet18

from utils.result_utils import average_data

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def str2bool(value):
    if isinstance(value, bool):
        return value

    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def run(args):

    time_list = []
    model_str = args.model

    for i in range(0, 1):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "MLR": # convex
            if "MNIST" in args.dataset:
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "CNN": # non-convex
            if "MNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "Omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "DNN": # non-convex
            if "MNIST" in args.dataset:
                args.model = DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNetP":
            args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            feature_dim = list(args.model.fc.parameters())[0].shape[1]
            args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "ResNet18":
            # args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

            args.model = ResNet18()
            args.model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
            args.model.fc = torch.nn.Linear(512, args.num_classes) # 将最后的全连接层改掉
            args.model = args.model.to(args.device)

            
            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)
        
        
        elif model_str == "ResNet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif model_str == "GoogleNet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False, 
                                                      num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif model_str == "LSTM":
            args.model = LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)


        elif model_str == "fastText":
            args.model = fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size, 
                                 num_classes=args.num_classes).to(args.device)
        
        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "HARCNN":
            if args.dataset == 'HAR':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                    pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'PAMAP2':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                    pool_kernel_size=(1, 2)).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)
        
        elif args.algorithm == "FedMK":
            server = FedMK(args, i)

        elif args.algorithm == "FedBulyan":
            server = FedBulyan(args, i)

        elif args.algorithm == "FedTrimmed":
            server = FedTrimmed(args, i)
            
        elif args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedMTL":
            server = FedMTL(args, i)

        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)

        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)

        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)

        elif args.algorithm == "APFL":
            server = APFL(args, i)

        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPer(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)

        elif args.algorithm == "FedPD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPD(args, i)


        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)

        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)


        elif args.algorithm == "FedSD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedSD(args, i)

        elif args.algorithm == "FedSDMK":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedSDMK(args, i)

        elif args.algorithm == "FedRT":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRepTrim(args, i)

        elif args.algorithm == "FedRTSD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRepTrimSD(args, i)

        elif args.algorithm == "FedFLIP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedFLIP(args, i)
            
        elif args.algorithm == "FedMedian":
            server = FedMedian(args, i)

        elif args.algorithm == "FedBulyan":
            server = FedBulyan(args, i)    
        
        elif args.algorithm == "APPLE":
            server = APPLE(args, i)
       
        elif args.algorithm == "FedCLCM":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedCLCM(args, i)

        else:
            raise NotImplementedError

        if args.attack == "badpfl":
            from attacks.badpfl_attack import BadPFLAttack
            BadPFLAttack.attach(server, args)
        elif args.attack == "pfedba":
            from attacks.pfedba_attack import PFedBAAttack
            PFedBAAttack.attach(server, args)

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times = 1)

    print("All done!")



if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=str2bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=str2bool, default=False)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=32000, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-ts', "--time_select", type=str2bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")

    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL / FedCAC
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    parser.add_argument('-lr_head', type=float, default=0.005,
                        help="head learning rate")
    # MOON / FedCAC / FedLC
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=str2bool, default=False)
    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)

    # FedDef
    parser.add_argument('--disti_epochs',  type=int, default=1)
    parser.add_argument('--protect_id',  type=int, default=1)

    # FedSD
    parser.add_argument('--adv_eps',  type=float, default=0.1)
    parser.add_argument('--adv_num_iter',  type=int, default=5)

    parser.add_argument('--defense_epochs',  type=int, default=1)
    parser.add_argument('--clean_data',  type=int, default=5000)

    # FedSDMK
    parser.add_argument('--sdmk_m',  type=int, default=7)
    parser.add_argument('--sdmk_k',  type=int, default=5)

    # FedPD
    # 🛠️ 在 args 中添加新参数
    parser.add_argument('--delta', type=float, default=0.5,  # 角度边界值
                    help='angular margin for contrastive loss')
    parser.add_argument('--sim_threshold', type=float, default=0.7,
                    help='similarity threshold for proto filtering')

    # FedRepTrim
    parser.add_argument('--rt_beta',  type=float, default=0.2)

    parser.add_argument('--num_adv_clients',  type=int, default=1)
    
    # FedCLCM
    parser.add_argument('--lambda_cl', type=float, default=0.5,
                        help='contrastive loss weight for FedCLCM')
    parser.add_argument('--aug_strength', type=float, default=0.1,
                        help='augmentation strength for contrastive learning')
    parser.add_argument('--mask_tau', type=float, default=2.0,
                        help='channel mask variance threshold multiplier')
    parser.add_argument('--mask_alpha', type=float, default=0.3,
                        help='channel mask downweight factor for suspicious channels')
    parser.add_argument('--enable_channel_mask', type=str2bool, default=True,
                        help='whether to enable channel masking')

    # FedCLCM - consistency gating & layer-wise trim
    parser.add_argument('--cosine_gate', action='store_true',
                        help='enable cosine gating for client updates')
    parser.add_argument('--cosine_gate_threshold', type=float, default=0.3,
                        help='cosine similarity threshold for gating')
    parser.add_argument('--cosine_gate_alpha', type=float, default=0.5,
                        help='downweight factor for low-similarity updates')
    parser.add_argument('--trim_high_layers', type=str, default="",
                        help='comma-separated layer prefixes for stronger trim')
    parser.add_argument('--trim_beta_high', type=float, default=None,
                        help='trim beta for high layers (if set)')
    parser.add_argument('--trim_beta_low', type=float, default=None,
                        help='trim beta for low layers (if set)')

    # Attack type (orthogonal to defense algorithm)
    parser.add_argument('--attack', type=str, default="none",
                        choices=["none", "badpfl", "pfedba"],
                        help='Attack type: none=use static dataset triggers, badpfl=Bad-PFL dynamic attack, pfedba=PFedBA gradient-matching attack')

    # Bad-PFL attack parameters (only used when --attack badpfl)
    parser.add_argument('--ba_target_label', type=int, default=0,
                        help='Bad-PFL: target label for the attack')
    parser.add_argument('--ba_poison_rate', type=float, default=0.2,
                        help='Bad-PFL: poison rate during training')
    parser.add_argument('--ba_trigger_gen_steps', type=int, default=30,
                        help='Bad-PFL: trigger generator training steps per round')
    parser.add_argument('--ba_trigger_gen_lr', type=float, default=1e-2,
                        help='Bad-PFL: trigger generator learning rate')
    parser.add_argument('--ba_pgd_eps', type=float, default=0.0314,
                        help='Bad-PFL: PGD epsilon (8/255 for [-1,1] range)')
    parser.add_argument('--ba_trigger_scale', type=float, default=8.0,
                        help='Bad-PFL: trigger perturbation scale (8.0 for [-1,1] range)')

    # PFedBA attack parameters (only used when --attack pfedba)
    parser.add_argument('--pfedba_target_label', type=int, default=0,
                        help='PFedBA: target label for the attack')
    parser.add_argument('--pfedba_poison_rate', type=float, default=0.5,
                        help='PFedBA: poison rate during training')
    parser.add_argument('--pfedba_poison_count', type=int, default=-1,
                        help='PFedBA: fixed poisoned samples per batch (original-code style), overrides poison_rate when > 0')
    parser.add_argument('--pfedba_poison_before_start', action='store_true',
                        help='PFedBA: if set, poison malicious client training even before attack_start')
    parser.add_argument('--pfedba_attack_start', type=int, default=30,
                        help='PFedBA: round to start trigger optimization')
    parser.add_argument('--pfedba_trigger_opt_steps', type=int, default=15,
                        help='PFedBA: gradient-matching optimization steps per round')
    parser.add_argument('--pfedba_trigger_lr', type=float, default=0.1,
                        help='PFedBA: trigger optimizer learning rate')
    parser.add_argument('--pfedba_init_value', type=float, default=0.5,
                        help='PFedBA: initial trigger value inside patch (original-code style uses non-zero init)')
    parser.add_argument('--pfedba_patch_size', type=int, default=8,
                        help='PFedBA: patch size (square), paper-style CIFAR-10 uses 8')
    parser.add_argument('--pfedba_patch_position', type=str, default="center",
                        choices=["center", "bottom_right"],
                        help='PFedBA: trigger patch position')
    parser.add_argument('--pfedba_img_c', type=int, default=3,
                        help='PFedBA: image channels')
    parser.add_argument('--pfedba_img_h', type=int, default=32,
                        help='PFedBA: image height')
    parser.add_argument('--pfedba_img_w', type=int, default=32,
                        help='PFedBA: image width')
    parser.add_argument('--pfedba_align_grad_weight', type=float, default=1.0,
                        help='PFedBA: weight for gradient alignment term')
    parser.add_argument('--pfedba_align_loss_weight', type=float, default=1.0,
                        help='PFedBA: weight for loss alignment term')
    parser.add_argument('--pfedba_attack_loss_weight', type=float, default=1.0,
                        help='PFedBA: weight for backdoor classification loss term')
    parser.add_argument('--pfedba_opt_samples_per_class', type=int, default=100,
                        help='PFedBA: trigger optimization samples per source class')
    parser.add_argument('--pfedba_opt_batch_size', type=int, default=64,
                        help='PFedBA: trigger optimization batch size')
    parser.add_argument('--pfedba_opt_max_batches_per_client', type=int, default=20,
                        help='PFedBA: max clean batches per malicious client when building optimization data')

    # Logit confidence analysis (clean vs backdoor)
    parser.add_argument('--logit_analysis', action='store_true',
                        help='analyze logit confidence for clean/backdoor')
    parser.add_argument('--logit_reject_thresholds', type=str, default="0.5,0.6,0.7,0.8",
                        help='comma-separated maxprob thresholds for rejection')
    parser.add_argument('--logit_entropy_thresholds', type=str, default="1.0,1.5,2.0",
                        help='comma-separated entropy thresholds for rejection')
    parser.add_argument('--logit_margin_thresholds', type=str, default="0.1,0.2,0.3",
                        help='comma-separated margin thresholds for rejection')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)

    run(args)
