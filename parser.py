import argparse


class Parser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--epochs", type=int, default=1,
                                 help="number of epochs of training")

        self.parser.add_argument("--data_dir", type=str, default='DEMAND_16KHz',
                                 help="name of the data_dir")
        self.parser.add_argument(
            "--csvfile", default='cut16128_ov0.5.csv', help="csv_file to use")
        self.parser.add_argument("--batch_size", type=int, default=2,
                                 help="size of the batches")
        self.parser.add_argument("--lr_G", type=float, default=0.0002,
                                 help="adam: generator learning rate")
        self.parser.add_argument("--lr_D", type=float, default=0.0004,
                                 help="adam: discriminator learning rate")
        self.parser.add_argument("--decay_epoch", type=int, default=25,
                                 help="epoch from which to start lr decay")
        self.parser.add_argument("--n_cpu", type=int, default=4,
                                 help="number of cpu threads to use during batch generation")
        self.parser.add_argument("--ngf", type=int, default=64,
                                 help="number of generator factor")
        self.parser.add_argument("--ndf", type=int, default=64,
                                 help="number of discriminator factor")
        self.parser.add_argument("--weights", type=list, default=[10, 100, 1, 20],
                                 help="metric_weight, mag_weight, istft_weight, phase_weight")
        self.parser.add_argument("--log_interval", type=int,
                                 default=1, help="log interval")
        self.parser.add_argument("--log_dir", type=str,
                                 default="./logs", help="log directory")
        self.parser.add_argument("--threshold", type=int,
                                 default=-80, help="threshold")
        self.parser.add_argument("--mode", default='train',
                                 help="initialize, train or test mode")
        self.parser.add_argument("--checkpt_dir", default='./checkpoints',
                                 help="the path to save model")
        self.parser.add_argument("--signal_minmax", default=False,
                                 help="use minmax normalization for 1d signal or not")
        self.parser.add_argument("--pesq_mode", default='own',
                                 help="use package pesq or our own pesq, 'pkg' or 'own'.")
        self.parser.add_argument("--num_experts", type=int, default='2',
                                 help="number of experts")
        self.parser.add_argument("--name", type=str,
                                 default='', help="experiment name")
        self.parser.add_argument("--model_for_init_experts", type=str,
                                 default='n_exp_2_bs_4_lre_0.0002_lrd_0.0004_e_1_1634244820', help="name of expert models")

    def parse_args(self):
        return self.parser.parse_args()
