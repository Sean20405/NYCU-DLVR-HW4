import argparse

parser = argparse.ArgumentParser()

# Device settings
parser.add_argument(
    '--device', type=int, default=0,
    help='GPU ID.'
)
parser.add_argument(
    '--num_workers', type=int, default=16,
    help='number of workers.'
)
parser.add_argument(
    "--num_gpus", type=int, default=1,
    help="Number of GPUs to use for training"
)

# Hyper-parameters
parser.add_argument(
    '--epochs', type=int, default=300,
    help='maximum number of epochs to train the total model.'
)
parser.add_argument(
    '--batch_size', type=int, default=4,
    help="Batch size to use per GPU"
)
parser.add_argument(
    '--lr', type=float, default=2e-4,
    help='learning rate of encoder.'
)
parser.add_argument(
    '--de_type', nargs='+', default=['derain', 'desnow'],
    help='which type of degradations is training and testing for.'
)
parser.add_argument(
    '--patch_size', type=int, default=128,
    help='patchsize of input.'
)

# Mode
parser.add_argument(
    '--train', action='store_true',
    help='train the model.'
)
parser.add_argument(
    '--test', action='store_true',
    help='test the model.'
)

# path
parser.add_argument(
    '--train_dir', type=str, default='data/train/',
    help='where the training data is stored.'
)
parser.add_argument(
    '--test_dir', type=str, default='data/test/degraded/',
    help='where the testing data is stored.'
)
parser.add_argument(
    '--output_path', type=str, default="results/",
    help='output save path'
)
parser.add_argument(
    '--wblogger', type=str, default="promptir",
    help="Determine to log to wandb or not and the project name"
)
parser.add_argument(
    '--ckpt_dir', type=str, default="ckpt",
    help="Name of the Directory where the checkpoint is to be saved"
)
parser.add_argument(
    '--ckpt_name', type=str,
    help='checkpoint load path'
)

options = parser.parse_args()
