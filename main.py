import logging
from parser import Parser
from utils import *
from tools.compute_metrics_norm import compute_metrics_norm
import warnings
from model.attention_unet_plusplus import AttUNetPlusPlus
from trainer import Trainer

from torchsummary import summary
from evaluation import Eval
from dataset import load_data
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')


def metrics(clean, noisy):
    clean = clean.numpy()
    noisy = noisy.numpy()
    mean_pesq_output_val_sliced, mean_csig_output_val_sliced, \
        mean_cbak_output_val_sliced, mean_covl_output_val_sliced, \
        mean_ssnr_output_val_sliced, mean_stoi_output_val_sliced = compute_metrics_norm(
            clean, noisy, 16000, 1, 0, 0)
    return mean_pesq_output_val_sliced, mean_csig_output_val_sliced, mean_cbak_output_val_sliced, \
        mean_covl_output_val_sliced, mean_ssnr_output_val_sliced, mean_stoi_output_val_sliced


def main():
    parser = Parser()
    opt = parser.parse_args()
    # Experiment name
    timestamp = str(int(time.time()))
    if opt.name == '':
        name = 'n_exp_{}_bs_{}_lre_{}_lrd_{}_e_{}_{}'.format(
            opt.num_experts, opt.batch_size, opt.lr_G, opt.lr_D,
            opt.epochs,
            timestamp)
        opt.name = name
    else:
        opt.name = '{}_{}'.format(opt.name, timestamp)
    print('\nExperiment: {}\n'.format(opt.name))
    print(opt)
    if not os.path.exists(opt.checkpt_dir):
        os.mkdir(opt.checkpt_dir)
    if not os.path.exists(opt.log_dir):
        os.mkdir(opt.log_dir)

    seed_torch(20)
    # device = [torch.device("cuda:0"), torch.device("cuda:1")]
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # device = torch.device("cpu")
    #available_gpus = [torch.cuda.get_device_name(
     #   i) for i in range(torch.cuda.device_count())]
    train_set, test_set = load_data(opt.data_dir,
                                    opt.csvfile, opt.batch_size, opt.threshold, opt.n_cpu)
    if opt.mode == 'initialize':
        trainer = Trainer(train_set, test_set, device, opt)
        trainer.initialize_experts()
    if opt.mode == 'train':
        trainer = Trainer(train_set, test_set, device, opt)
        trainer.train()
    elif opt.mode == 'test':
        model = AttUNetPlusPlus(in_channel=3, ngf=opt.ngf).cuda()
        model.load_state_dict(torch.load(
            "/no_backups/s1374/PMAGAN-1/saved/_new2_27"))
        model.eval()
        evaluator = Eval(model, test_set, opt.threshold, opt.batch_size)
        evaluator.test()


if __name__ == "__main__":
    main()
