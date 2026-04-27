"""FedCLCMPurify server for the PFedBA stack."""

from FLAlgorithms.servers.serverfedclcm import ServerFedCLCM
from FLAlgorithms.users.userclcmpurify import UserCLCMPurify


class ServerFedCLCMPurify(ServerFedCLCM):
    user_class = UserCLCMPurify

    def __init__(
        self,
        *args,
        purify_beta=0.0,
        purify_feature_beta=0.0,
        purify_logit_beta=0.0,
        purify_temperature=2.0,
        purify_start_round=1,
        purify_layers="layer4",
        purify_teacher_momentum=0.9,
        purify_teacher_cpu_half=True,
        **kwargs,
    ):
        self.extra_user_kwargs = {
            "purify_beta": purify_beta,
            "purify_feature_beta": purify_feature_beta,
            "purify_logit_beta": purify_logit_beta,
            "purify_temperature": purify_temperature,
            "purify_start_round": purify_start_round,
            "purify_layers": purify_layers,
            "purify_teacher_momentum": purify_teacher_momentum,
            "purify_teacher_cpu_half": purify_teacher_cpu_half,
        }
        super().__init__(*args, **kwargs)
        print(
            "Finished creating FedCLCMPurify (PFedBA) server: "
            f"beta={purify_beta}, feature_beta={purify_feature_beta}, "
            f"logit_beta={purify_logit_beta}, layers={purify_layers}, "
            f"start={purify_start_round}, momentum={purify_teacher_momentum}"
        )
