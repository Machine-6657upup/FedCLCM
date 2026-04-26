import copy
import random
import argparse
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverprox import FedProx
from FLAlgorithms.servers.serverrep import FedRep
from FLAlgorithms.servers.serverfedrt import ServerFedRT
from FLAlgorithms.servers.serverfedclcm import ServerFedCLCM
from FLAlgorithms.servers.serverfedrpd import ServerFedRPD
from FLAlgorithms.servers.serverpflalp import ServerPFLALP
from FLAlgorithms.servers.serverbdpfl import ServerBDPFL
from FLAlgorithms.servers.serverfedreppflalp import ServerFedRepPFLALP
from FLAlgorithms.servers.serverfedrepbdpfl import ServerFedRepBDPFL
from FLAlgorithms.trainmodel.mnist_model import MnistNet
from FLAlgorithms.trainmodel.fashionmnist_model import FMnistNet
from FLAlgorithms.trainmodel.models import CifarNet
from FLAlgorithms.trainmodel.cifar_model import ResNet18_cifar, ResNet18_cifar_pretrained
from FLAlgorithms.trainmodel.fedrep_model import to_fedrep_model
from utils.model_utils import read_data
from utils.plot_utils import *
import torch
import datetime


def set_global_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


set_global_seed(1)


LEGACY_FIXED_MALCLIENT_IDS = [
    'f_00007', 'f_00001', 'f_00062', 'f_00020', 'f_00096',
    'f_00085', 'f_00051', 'f_00043', 'f_00037', 'f_00058',
]


def build_malicious_clients(dataset, malnum, malclient_id_mode, seed=1):
    data = read_data(dataset)
    client_ids = list(data[0])
    if len(client_ids) == 0:
        return []

    malnum = max(0, min(int(malnum), len(client_ids)))
    if malnum == 0:
        return []

    rng = np.random.RandomState(int(seed))
    if malclient_id_mode == "legacy_fixed":
        chosen = []
        for client_id in LEGACY_FIXED_MALCLIENT_IDS:
            if client_id in client_ids and client_id not in chosen:
                chosen.append(client_id)
            if len(chosen) >= malnum:
                return chosen

        remaining = [cid for cid in client_ids if cid not in chosen]
        if len(chosen) < malnum:
            extra = rng.choice(np.array(remaining, dtype=object), size=malnum - len(chosen), replace=False).tolist()
            chosen.extend(extra)
        return chosen

    if malclient_id_mode == "seeded_pool":
        return rng.choice(np.array(client_ids, dtype=object), size=malnum, replace=False).tolist()

    raise ValueError("malclient_id_mode wrong!")


def main(
    dataset,
    algorithm,
    model,
    batch_size,
    learning_rate,
    beta,
    lamda,
    num_glob_iters,
    local_epochs,
    optimizer,
    numusers,
    K,
    personal_learning_rate,
    times,
    malnum,
    poisonratio,
    attack_method,
    per_epoch,
    attack_start,
    oneshot,
    clip_rate,
    defense,
    resnet_pretrained,
    lr_head=0.1,
    plocal_epochs=1,
    rt_beta=0.2,
    mask_tau=2.0,
    mask_alpha=0.3,
    enable_channel_mask=1,
    lambda_cl=0.5,
    aug_strength=0.1,
    adv_eps=0.0,
    adv_num_iter=0,
    purify_beta=1500.0,
    purify_rounds=1,
    cluster_max_k=4,
    alp_use_cluster=1,
    alp_use_purify=1,
    bd_lambda=1.0,
    bd_tau=1.0,
    bd_gamma=1.0,
    bd_use_inter=1,
    bd_use_em=1,
    distill_gamma=1.0,
    distill_weight=1.0,
    cosine_gate=0,
    cosine_gate_threshold=0.3,
    cosine_gate_alpha=0.5,
    trim_high_layers="",
    trim_beta_high=None,
    trim_beta_low=None,
    lr_decay=1.0,
    lr_decay_step=0,
    selection_strategy="uniform",
    fixed_malicious_per_round=0,
    malclient_id_mode="legacy_fixed",
    mal_local_epoch=20,
    mal_learning_rate=None,
    eval_gap=1,
    personalized_eval_gap=0,
    seed=1,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = model

    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

    trigger_patten = []
    trigger_list = []

    # load trigger
    if dataset == 'Mnist' or dataset == 'FashionMnist':
        for i in range(10, 20, 1):
            for j in range(10, 20, 1):
                trigger_patten.append([i, j])

        malclient = build_malicious_clients(dataset, malnum, malclient_id_mode, seed=seed)
        poison_label = 1
        intinal_trigger = torch.zeros((1, 28, 28)).float().to(device)

        for i in trigger_patten:
            intinal_trigger[0][i[0]][i[1]] = 0.5

        for i in range(10):
            trigger_list.append(copy.deepcopy(intinal_trigger))

    elif dataset == 'Cifar10':
        for i in range(12, 20, 1):
            for j in range(12, 20, 1):
                trigger_patten.append([i, j])
        malclient = build_malicious_clients(dataset, malnum, malclient_id_mode, seed=seed)
        poison_label = 0
        intinal_trigger = torch.zeros((3, 32, 32)).float().to(device)
        for i in trigger_patten:
            intinal_trigger[0][i[0]][i[1]] = 0.5
            intinal_trigger[1][i[0]][i[1]] = 0.5
            intinal_trigger[2][i[0]][i[1]] = 0.5
        for _ in range(10):
            trigger_list.append(copy.deepcopy(intinal_trigger))
    else:
        raise ValueError("dataset name wrong!")

    print(malclient)

    for i in range(times):  # 重复实验
        run_seed = int(seed) + i
        set_global_seed(run_seed)
        print("---------------Running time:------------", i, "seed:", run_seed)
        # Generate model

        if dataset == 'Mnist':
            model = MnistNet(name="global", created_time=current_time).to(device)

        elif dataset == 'FashionMnist':
            model = FMnistNet(name="global", created_time=current_time).to(device)
        elif dataset == 'Cifar10':
            if model_name.lower() == 'resnet':
                if int(resnet_pretrained) == 1:
                    model = ResNet18_cifar_pretrained(name="global", created_time=current_time).to(device)
                else:
                    model = ResNet18_cifar(name="global", created_time=current_time).to(device)
            else:
                model = CifarNet().to(device)

        else:
            raise ValueError("dataset name wrong!")

        if algorithm in ("FedRep", "FedRT", "FedCLCM", "FedRPD", "FedRepPFLALP", "FedRepBDPFL"):
            model = to_fedrep_model(model)

        # select algorithm
        if algorithm == "FedAvg":
            server = FedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                            local_epochs, optimizer, numusers, i, False, current_time=current_time, malnum=malnum,
                            malclient=malclient, poisonratio=poisonratio, poison_label=poison_label,
                            attack_method=attack_method, per_epoch=per_epoch, defense=defense)

        elif algorithm == "FedProx":
            server = FedProx(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                             local_epochs, optimizer, numusers, i, False, current_time=current_time, malnum=malnum,
                             malclient=malclient, poisonratio=poisonratio, poison_label=poison_label,
                             attack_method=attack_method, per_epoch=per_epoch, defense=defense)
        elif algorithm == "FedRep":
            server = FedRep(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                            local_epochs, optimizer, numusers, i, False, current_time=current_time, malnum=malnum,
                            malclient=malclient, poisonratio=poisonratio, poison_label=poison_label,
                            attack_method=attack_method, per_epoch=per_epoch, defense=defense,
                            lr_head=lr_head, plocal_epochs=plocal_epochs)
        elif algorithm == "FedRT":
            server = ServerFedRT(
                device,
                dataset,
                algorithm,
                model,
                batch_size,
                learning_rate,
                beta,
                lamda,
                num_glob_iters,
                local_epochs,
                optimizer,
                numusers,
                i,
                False,
                current_time=current_time,
                malnum=malnum,
                malclient=malclient,
                poisonratio=poisonratio,
                poison_label=poison_label,
                attack_method=attack_method,
                per_epoch=per_epoch,
                defense=defense,
                lr_head=lr_head,
                plocal_epochs=plocal_epochs,
                rt_beta=rt_beta,
                aug_strength=aug_strength,
                adv_eps=adv_eps,
                adv_num_iter=int(adv_num_iter),
            )
        elif algorithm == "FedRPD":
            server = ServerFedRPD(
                device,
                dataset,
                algorithm,
                model,
                batch_size,
                learning_rate,
                beta,
                lamda,
                num_glob_iters,
                local_epochs,
                optimizer,
                numusers,
                i,
                False,
                current_time=current_time,
                malnum=malnum,
                malclient=malclient,
                poisonratio=poisonratio,
                poison_label=poison_label,
                attack_method=attack_method,
                per_epoch=per_epoch,
                defense=defense,
                lr_head=lr_head,
                plocal_epochs=plocal_epochs,
                rt_beta=rt_beta,
                aug_strength=aug_strength,
                adv_eps=adv_eps,
                adv_num_iter=int(adv_num_iter),
                purify_beta=purify_beta,
                purify_rounds=int(purify_rounds),
                distill_gamma=distill_gamma,
                distill_weight=distill_weight,
            )
        elif algorithm == "PFLALP":
            server = ServerPFLALP(
                device,
                dataset,
                algorithm,
                model,
                batch_size,
                learning_rate,
                beta,
                lamda,
                num_glob_iters,
                local_epochs,
                optimizer,
                numusers,
                i,
                False,
                current_time=current_time,
                malnum=malnum,
                malclient=malclient,
                poisonratio=poisonratio,
                poison_label=poison_label,
                attack_method=attack_method,
                per_epoch=per_epoch,
                defense=defense,
                lr_head=lr_head,
                plocal_epochs=plocal_epochs,
                purify_beta=purify_beta,
                purify_rounds=int(purify_rounds),
                cluster_max_k=int(cluster_max_k),
                lr_decay=lr_decay,
                lr_decay_step=int(lr_decay_step),
                mal_local_epochs=int(mal_local_epoch),
                mal_learning_rate=mal_learning_rate,
            )
        elif algorithm == "FedRepPFLALP":
            server = ServerFedRepPFLALP(
                device,
                dataset,
                algorithm,
                model,
                batch_size,
                learning_rate,
                beta,
                lamda,
                num_glob_iters,
                local_epochs,
                optimizer,
                numusers,
                i,
                False,
                current_time=current_time,
                malnum=malnum,
                malclient=malclient,
                poisonratio=poisonratio,
                poison_label=poison_label,
                attack_method=attack_method,
                per_epoch=per_epoch,
                defense=defense,
                lr_head=lr_head,
                plocal_epochs=plocal_epochs,
                alp_use_cluster=alp_use_cluster,
                alp_use_purify=alp_use_purify,
                purify_beta=purify_beta,
                purify_rounds=int(purify_rounds),
                cluster_max_k=int(cluster_max_k),
            )
        elif algorithm == "BDPFL":
            server = ServerBDPFL(
                device,
                dataset,
                algorithm,
                model,
                batch_size,
                learning_rate,
                beta,
                lamda,
                num_glob_iters,
                local_epochs,
                optimizer,
                numusers,
                i,
                False,
                current_time=current_time,
                malnum=malnum,
                malclient=malclient,
                poisonratio=poisonratio,
                poison_label=poison_label,
                attack_method=attack_method,
                per_epoch=per_epoch,
                defense=defense,
                bd_lambda=bd_lambda,
                bd_tau=bd_tau,
                bd_gamma=bd_gamma,
                bd_use_inter=bd_use_inter,
                bd_use_em=bd_use_em,
                lr_decay=lr_decay,
                lr_decay_step=int(lr_decay_step),
            )
        elif algorithm == "FedRepBDPFL":
            server = ServerFedRepBDPFL(
                device,
                dataset,
                algorithm,
                model,
                batch_size,
                learning_rate,
                beta,
                lamda,
                num_glob_iters,
                local_epochs,
                optimizer,
                numusers,
                i,
                False,
                current_time=current_time,
                malnum=malnum,
                malclient=malclient,
                poisonratio=poisonratio,
                poison_label=poison_label,
                attack_method=attack_method,
                per_epoch=per_epoch,
                defense=defense,
                lr_head=lr_head,
                plocal_epochs=plocal_epochs,
                bd_lambda=bd_lambda,
                bd_tau=bd_tau,
                bd_gamma=bd_gamma,
                bd_use_inter=bd_use_inter,
                bd_use_em=bd_use_em,
            )

        elif algorithm == "FedCLCM":
            server = ServerFedCLCM(
                device,
                dataset,
                algorithm,
                model,
                batch_size,
                learning_rate,
                beta,
                lamda,
                num_glob_iters,
                local_epochs,
                optimizer,
                numusers,
                i,
                False,
                current_time=current_time,
                malnum=malnum,
                malclient=malclient,
                poisonratio=poisonratio,
                poison_label=poison_label,
                attack_method=attack_method,
                per_epoch=per_epoch,
                defense=defense,
                lr_head=lr_head,
                plocal_epochs=plocal_epochs,
                rt_beta=rt_beta,
                mask_tau=mask_tau,
                mask_alpha=mask_alpha,
                enable_channel_mask=bool(enable_channel_mask),
                lambda_cl=lambda_cl,
                aug_strength=aug_strength,
                adv_eps=adv_eps,
                adv_num_iter=int(adv_num_iter),
                cosine_gate=bool(cosine_gate),
                cosine_gate_threshold=cosine_gate_threshold,
                cosine_gate_alpha=cosine_gate_alpha,
                trim_high_layers=trim_high_layers or None,
                trim_beta_high=trim_beta_high,
                trim_beta_low=trim_beta_low,
            )

        else:
            raise ValueError("alg name wrong!")

        server.eval_gap = max(int(eval_gap), 1)
        server.personalized_eval_gap = max(int(personalized_eval_gap), 0)
        server.selection_strategy = selection_strategy
        server.fixed_malicious_per_round = max(int(fixed_malicious_per_round), 0)

        final_trigger_list = server.train(pattern=trigger_patten, trigger=trigger_list, per_epoch=per_epoch,
                                          attack_start=attack_start, oneshot=oneshot, clip_rate=clip_rate,
                                          defense=defense)

        print(final_trigger_list[0])

        # local finetuning
        server.send_parameters()  # 将当前的全局模型分给每一个用户 deepcopy

        # Evaluate the final global model
        print("Evaluate the final global model")
        globaltestasr, globaltrainasr, globaltrainasrloss, global_test_mean_benign_acc, global_test_mean_mal_acc = server.evaluate()  # 分发了模型，当前用户的模型是全局模型， 输出所有本地数据在全局模型上测试的准确率
        globaltestasr, globaltrainasr, globaltrainasrloss, global_test_mean_benign_asr, global_test_mean_mal_asr = server.poison_evaluate(
            trigger=final_trigger_list, pattern=trigger_patten)  # 输出所有本地数据在全局模型上测试的准确率

        print("")

        # Evaluate gloal model on user for each interation
        print("Evaluate the final global model with a few step update, which is personalized model")
        server.send_parameters()
        pertestacc, pertrainacc, pertrainloss, poiperasr, poipertrainasr, poiperloss, per_mean_ben_asr, per_mean_mal_asr, per_mean_ben_acc, per_mean_mal_acc = server.evaluate_one_step(
            per_epoch, trigger=final_trigger_list, pattern=trigger_patten)  # 本地finetune一次，在本地模型上再测试   --个性化模型准确率

        if algorithm in ("PFLALP", "FedRepPFLALP"):
            print("")
            print("Evaluate saved personalized models for PFL-ALP style inference")
            server.evaluate_personalized_model()
            server.evaluate_personalized_model_poison(trigger=final_trigger_list, pattern=trigger_patten)
        if algorithm in ("BDPFL", "FedRepBDPFL"):
            print("")
            print("Evaluate saved personalized models for BDPFL style inference")
            server.evaluate_personalized_model()
            server.evaluate_personalized_model_poison(trigger=final_trigger_list, pattern=trigger_patten)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FashionMnist",
                        choices=["Mnist", "FashionMnist", "Cifar10"])
    parser.add_argument("--model", type=str, default="cnn", choices=["dnn", "mclr", "cnn", "VGG16", "resnet", "lenet"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedMe",
                        choices=["pFedMe", "PerAvg-FO", "PerAvg-HF", "FedAvg", "FedProx", "FedRep", "FedRT", "FedCLCM", "FedRPD", "PFLALP", "BDPFL", "FedRepPFLALP", "FedRepBDPFL", "Ditto", "SCAFFOLD", "FedBN"])
    parser.add_argument("--numusers", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.09,
                        help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--seed", type=int, default=1, help="base random seed; repeated runs use seed+i")
    parser.add_argument("--malclient", type=int, default=10, help="number of malicious client")
    parser.add_argument("--attack_start", type=int, default=10, help="the start attack iteration")
    parser.add_argument("--mal_local_epoch", type=int, default=20)
    parser.add_argument("--poisoning_per_batch", type=int, default=5, help="the poison ratio")
    parser.add_argument("--attack_method", type=str, default='attackall',
                        choices=['attackall'])
    parser.add_argument("--attack_goal", type=str, default='attackall', choices=['attackone', 'attackall'])
    parser.add_argument("--per_epoch", type=int, default='1', help='the epoch for local finetune')
    parser.add_argument("--plocal_epochs", type=int, default=1, help="FedRep head-update epochs per round")
    parser.add_argument("--lr_head", type=float, default=0.1, help="FedRep head learning rate")
    parser.add_argument("--descrip", type=str, help="the gradient mask ratio")
    parser.add_argument("--oneshot", type=int, default=0, help="one shot attack", choices=[1, 0])
    parser.add_argument("--clip_rate", type=int, default=0, help="one shot attack scale")
    parser.add_argument("--defense", type=str, default='none', help="defense method",
                        choices=['none', 'mkrum', 'trim'])
    parser.add_argument("--resnet_pretrained", type=int, default=0, choices=[0, 1],
                        help="whether to initialize CIFAR ResNet with ImageNet pretrained weights")
    # FedCLCM (ported from root project; use --algorithm FedCLCM)
    parser.add_argument("--rt_beta", type=float, default=0.2, help="trim ratio for FedCLCM trimmed-mean on base")
    parser.add_argument("--mask_tau", type=float, default=2.0, help="channel variance threshold multiplier")
    parser.add_argument("--mask_alpha", type=float, default=0.3, help="down-weight for suspicious channels")
    parser.add_argument("--enable_channel_mask", type=int, default=1, choices=[0, 1])
    parser.add_argument("--lambda_cl", type=float, default=0.5, help="contrastive loss weight (benign clients)")
    parser.add_argument("--aug_strength", type=float, default=0.1, help="noise/cutout strength for CL & trigger breaking")
    parser.add_argument("--adv_eps", type=float, default=0.0, help="PGD epsilon on head phase (benign); 0 disables")
    parser.add_argument("--adv_num_iter", type=int, default=0, help="PGD iterations; 0 disables")
    parser.add_argument("--purify_beta", type=float, default=1500.0, help="PFL-ALP style benign purification strength")
    parser.add_argument("--purify_rounds", type=int, default=1, help="number of benign purification passes after local training")
    parser.add_argument("--cluster_max_k", type=int, default=4, help="max cluster count tried by PFL-ALP dynamic clustering")
    parser.add_argument("--alp_use_cluster", type=int, default=1, choices=[0, 1], help="enable PFL-ALP cluster representative module on FedRep")
    parser.add_argument("--alp_use_purify", type=int, default=1, choices=[0, 1], help="enable PFL-ALP NAD purification module on FedRep")
    parser.add_argument("--bd_lambda", type=float, default=1.0, help="BDPFL mutual distillation weight")
    parser.add_argument("--bd_tau", type=float, default=1.0, help="BDPFL distillation temperature")
    parser.add_argument("--bd_gamma", type=float, default=1.0, help="BDPFL layer weight decay gamma")
    parser.add_argument("--bd_use_inter", type=int, default=1, choices=[0, 1], help="enable BDPFL layer-wise feature distillation")
    parser.add_argument("--bd_use_em", type=int, default=1, choices=[0, 1], help="enable BDPFL explanation heatmap distillation")
    parser.add_argument("--distill_gamma", type=float, default=1.0, help="BDPFL style layer-weight decay factor")
    parser.add_argument("--distill_weight", type=float, default=1.0, help="weight of layer-wise feature distillation")
    parser.add_argument("--cosine_gate", type=int, default=0, choices=[0, 1])
    parser.add_argument("--cosine_gate_threshold", type=float, default=0.3)
    parser.add_argument("--cosine_gate_alpha", type=float, default=0.5)
    parser.add_argument("--trim_high_layers", type=str, default="", help="comma-separated name prefixes for stronger trim")
    parser.add_argument("--trim_beta_high", type=float, default=None, help="trim rt_beta override on trim_high_layers (FedCLCM)")
    parser.add_argument("--trim_beta_low", type=float, default=None, help="trim rt_beta override on other layers (FedCLCM)")
    parser.add_argument("--lr_decay", type=float, default=1.0, help="multiply lr by this factor every lr_decay_step rounds")
    parser.add_argument("--lr_decay_step", type=int, default=0, help="round interval for lr decay; <=0 disables")
    parser.add_argument("--selection_strategy", type=str, default="uniform", choices=["uniform", "fixed_malicious_mix"])
    parser.add_argument("--fixed_malicious_per_round", type=int, default=0, help="for fixed_malicious_mix, selected malicious clients each round")
    parser.add_argument("--malclient_id_mode", type=str, default="legacy_fixed", choices=["legacy_fixed", "seeded_pool"])
    parser.add_argument("--mal_learning_rate", type=float, default=None, help="optional attacker local lr override for full-model methods")
    parser.add_argument("--eval_gap", type=int, default=1, help="round gap for periodic global evaluation during training")
    parser.add_argument("--personalized_eval_gap", type=int, default=0, help="round gap for periodic personalized evaluation during training; 0 disables and keeps only final personalized evaluation")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Attack method:{}".format(args.attack_method))
    print("Defense method:{}".format(args.defense))
    print("Attack goal:{}".format(args.attack_goal))
    print("Start attack iteration:{}".format(args.attack_start))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Random seed        : {}".format(args.seed))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("ResNet Pretrained : {}".format(args.resnet_pretrained))
    print("Malicious client id mode: {}".format(args.malclient_id_mode))
    print("Selection strategy: {} / fixed malicious per round: {}".format(
        args.selection_strategy, args.fixed_malicious_per_round))
    print("LR decay          : {} every {} rounds".format(args.lr_decay, args.lr_decay_step))
    print("Eval gap          : {} / personalized eval gap: {}".format(
        args.eval_gap, args.personalized_eval_gap))
    if args.algorithm == "FedRT":
        print("FedRT rt_beta={} adv_eps={} adv_num_iter={} aug_strength={}".format(
            args.rt_beta, args.adv_eps, args.adv_num_iter, args.aug_strength))
    if args.algorithm == "FedRPD":
        print("FedRPD rt_beta={} adv_eps={} adv_num_iter={} aug_strength={} purify_beta={} purify_rounds={} distill_gamma={} distill_weight={}".format(
            args.rt_beta, args.adv_eps, args.adv_num_iter, args.aug_strength,
            args.purify_beta, args.purify_rounds, args.distill_gamma, args.distill_weight))
    if args.algorithm == "PFLALP":
        print("PFL-ALP-full purify_beta={} purify_rounds={} cluster_max_k={} mal_local_epoch={} mal_learning_rate={}".format(
            args.purify_beta, args.purify_rounds, args.cluster_max_k, args.mal_local_epoch, args.mal_learning_rate))
    if args.algorithm == "FedRepPFLALP":
        print("FedRepPFLALP alp_use_cluster={} alp_use_purify={} purify_beta={} purify_rounds={} cluster_max_k={}".format(
            args.alp_use_cluster, args.alp_use_purify, args.purify_beta, args.purify_rounds, args.cluster_max_k))
    if args.algorithm == "BDPFL":
        print("BDPFL bd_lambda={} bd_tau={} bd_gamma={} bd_use_inter={} bd_use_em={}".format(
            args.bd_lambda, args.bd_tau, args.bd_gamma, args.bd_use_inter, args.bd_use_em))
    if args.algorithm == "FedRepBDPFL":
        print("FedRepBDPFL bd_lambda={} bd_tau={} bd_gamma={} bd_use_inter={} bd_use_em={}".format(
            args.bd_lambda, args.bd_tau, args.bd_gamma, args.bd_use_inter, args.bd_use_em))
    if args.algorithm == "FedCLCM":
        print("FedCLCM rt_beta={} mask_tau={} lambda_cl={} channel_mask={}".format(
            args.rt_beta, args.mask_tau, args.lambda_cl, args.enable_channel_mask))
    print("Per_local epoch:{}".format(args.per_epoch))
    print("one shot:{}".format(args.oneshot))
    print("scale rate:{}".format(args.clip_rate))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm=args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        lamda=args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer=args.optimizer,
        numusers=args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times=args.times,
        malnum=args.malclient,
        poisonratio=args.poisoning_per_batch,
        attack_method=args.attack_method,
        per_epoch=args.per_epoch,
        attack_start=args.attack_start,
        oneshot=args.oneshot,
        clip_rate=args.clip_rate,
        defense=args.defense,
        resnet_pretrained=args.resnet_pretrained,
        lr_head=args.lr_head,
        plocal_epochs=args.plocal_epochs,
        rt_beta=args.rt_beta,
        mask_tau=args.mask_tau,
        mask_alpha=args.mask_alpha,
        enable_channel_mask=args.enable_channel_mask,
        lambda_cl=args.lambda_cl,
        aug_strength=args.aug_strength,
        adv_eps=args.adv_eps,
        adv_num_iter=args.adv_num_iter,
        purify_beta=args.purify_beta,
        purify_rounds=args.purify_rounds,
        cluster_max_k=args.cluster_max_k,
        alp_use_cluster=args.alp_use_cluster,
        alp_use_purify=args.alp_use_purify,
        bd_lambda=args.bd_lambda,
        bd_tau=args.bd_tau,
        bd_gamma=args.bd_gamma,
        bd_use_inter=args.bd_use_inter,
        bd_use_em=args.bd_use_em,
        distill_gamma=args.distill_gamma,
        distill_weight=args.distill_weight,
        cosine_gate=args.cosine_gate,
        cosine_gate_threshold=args.cosine_gate_threshold,
        cosine_gate_alpha=args.cosine_gate_alpha,
        trim_high_layers=args.trim_high_layers,
        trim_beta_high=args.trim_beta_high,
        trim_beta_low=args.trim_beta_low,
        lr_decay=args.lr_decay,
        lr_decay_step=args.lr_decay_step,
        selection_strategy=args.selection_strategy,
        fixed_malicious_per_round=args.fixed_malicious_per_round,
        malclient_id_mode=args.malclient_id_mode,
        mal_local_epoch=args.mal_local_epoch,
        mal_learning_rate=args.mal_learning_rate,
        eval_gap=args.eval_gap,
        personalized_eval_gap=args.personalized_eval_gap,
        seed=args.seed,
    )
