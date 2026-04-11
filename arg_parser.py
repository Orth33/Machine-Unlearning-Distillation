import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Lottery Tickets Experiments")

    ##################################### Dataset #################################################
    parser.add_argument(
        "--data", type=str, default="../data", help="location of the data corpus"
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument(
        "--input_size", type=int, default=32, help="size of input images"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./tiny-imagenet-200",
        help="dir to tiny-imagenet",
    )
    parser.add_argument(
        "--ham_metadata",
        type=str,
        default=None,
        help="optional path to HAM10000_metadata.csv",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)
    ##################################### Architecture ############################################
    parser.add_argument(
        "--arch", type=str, default="resnet18", help="model architecture"
    )
    parser.add_argument(
        "--imagenet_arch",
        action="store_true",
        help="architecture for imagenet size samples",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="load torchvision pretrained weights when supported",
    )
    parser.add_argument(
        "--train_y_file",
        type=str,
        default="./labels/train_ys.pth",
        help="labels for training files",
    )
    parser.add_argument(
        "--val_y_file",
        type=str,
        default="./labels/val_ys.pth",
        help="labels for validation files",
    )
    ##################################### General setting ############################################
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument(
        "--train_seed",
        default=1,
        type=int,
        help="seed for training (default value same as args.seed)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers in dataloader"
    )
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
    parser.add_argument(
        "--save_dir",
        help="The directory used to save the trained models",
        default=None,
        type=str,
    )
    parser.add_argument("--mask", type=str, default=None, help="sparse model")

    ##################################### Training setting #################################################
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
        help="limit training to the first N mini-batches per epoch; 0 disables the cap",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=0,
        help="limit evaluation to the first N mini-batches; 0 disables the cap",
    )
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument(
        "--train_label_smoothing",
        default=0.0,
        type=float,
        help="label smoothing epsilon for baseline training",
    )
    parser.add_argument(
        "--class_balanced_loss",
        action="store_true",
        help="use inverse-frequency class weights in baseline training",
    )
    parser.add_argument(
        "--epochs", default=182, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--warmup", default=0, type=int, help="warm up epochs")
    parser.add_argument("--print_freq", default=50, type=int, help="print frequency")
    parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")
    parser.add_argument(
        "--no-aug",
        action="store_true",
        default=False,
        help="No augmentation in training dataset (transformation).",
    )
    parser.add_argument("--no-l1-epochs", default=0, type=int, help="non l1 epochs")
    ##################################### Pruning setting #################################################
    parser.add_argument("--prune", type=str, default="omp", help="method to prune")
    parser.add_argument(
        "--pruning_times",
        default=1,
        type=int,
        help="overall times of pruning (only works for IMP)",
    )
    parser.add_argument(
        "--rate", default=0.95, type=float, help="pruning rate"
    )  # pruning rate is always 20%
    parser.add_argument(
        "--prune_type",
        default="rewind_lt",
        type=str,
        help="IMP type (lt, pt or rewind_lt)",
    )
    parser.add_argument(
        "--random_prune", action="store_true", help="whether using random prune"
    )
    parser.add_argument("--rewind_epoch", default=0, type=int, help="rewind checkpoint")
    parser.add_argument(
        "--rewind_pth", default=None, type=str, help="rewind checkpoint to load"
    )

    ##################################### Unlearn setting #################################################
    parser.add_argument(
        "--unlearn", type=str, default="retrain", help="method to unlearn"
    )
    parser.add_argument(
        "--unlearn_lr", default=0.01, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--unlearn_epochs",
        default=10,
        type=int,
        help="number of total epochs for unlearn to run",
    )
    parser.add_argument(
        "--num_indexes_to_replace",
        type=int,
        default=None,
        help="Number of data to forget",
    )
    parser.add_argument(
        "--class_to_replace", type=int, default=-1, help="Specific class to forget"
    )
    parser.add_argument(
        "--indexes_to_replace",
        type=list,
        default=None,
        help="Specific index data to forget",
    )
    parser.add_argument("--alpha", default=0.2, type=float, help="unlearn noise")

    # ----------------------- Minimal FT improvements -----------------------
    parser.add_argument("--head_reset", action="store_true",
                        help="If set and the forget set contains a single class (classwise forgetting), zero/re-init that class's classifier row before unlearning. Helps drive forget accuracy to ~0 while preserving other classes.")
    parser.add_argument("--freeze_bn", action="store_true",
                        help="Freeze BatchNorm running stats and affine params during unlearning to avoid distribution drift. Recommended when using GA/FT with small forget sets.")

    # ----------------------- Unlearning+Repair extras -----------------------
    parser.add_argument("--retain_steps", type=int, default=2,
                        help="Number of retain (repair) mini-batches per epoch used after each forget-ascent pass.")
    parser.add_argument("--forget_steps", type=int, default=1,
                        help="Number of forget (gradient ascent) mini-batches per epoch.")
    parser.add_argument("--mixup_alpha", type=float, default=0.2,
                        help="Mixup alpha for retain-side repair; 0 disables Mixup.")
    parser.add_argument("--smoothing_eps", type=float, default=0.05,
                        help="Label smoothing epsilon used in retain-side repair.")
    parser.add_argument("--anchor_lambda", type=float, default=1e-4,
                        help="L2 parameter anchoring strength towards the pre-unlearning sparse checkpoint.")
    parser.add_argument("--entropy_beta", type=float, default=0.5,
                        help="Weight for entropy maximization on forget batches (raises uncertainty on forgotten data).")
    parser.add_argument("--repair_lr", type=float, default=0.01,
                        help="Learning rate for the retain-side SAM optimizer.")
    parser.add_argument("--sam_rho", type=float, default=2.0,
                        help="SAM neighborhood size (rho) for retain-side repair.")
    parser.add_argument("--bn_recalibrate_batches", type=int, default=256,
                        help="Number of retain batches to use to recalibrate BatchNorm stats after unlearning; 0 disables.")
    parser.add_argument(
        "--skip_mia",
        action="store_true",
        help="skip MIA evaluation during smoke tests or fast debugging runs",
    )

    # === FT_conf parameters ===
    parser.add_argument("--forget_temperature", type=float, default=2.0,
                        help="Temperature used for complementary cross-entropy on forget batches.")
    parser.add_argument("--forget_margin", type=float, default=1.0,
                        help="Margin for pushing down the forgotten-class logit.")
    parser.add_argument("--forget_weight", type=float, default=1.0,
                        help="Weight for forget-side objective.")
    parser.add_argument("--retain_weight", type=float, default=1.0,
                        help="Weight for retain-side objective.")
    parser.add_argument("--steps_per_retain", type=int, default=2,
                        help="How many retain updates to perform after each forget update.")
    
    ##################################### Attack setting #################################################
    parser.add_argument(
        "--attack", type=str, default="backdoor", help="method to unlearn"
    )
    parser.add_argument(
        "--trigger_size",
        type=int,
        default=4,
        help="The size of trigger of backdoor attack",
    )
    return parser.parse_args()
