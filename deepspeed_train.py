import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import deepspeed
import json
import numpy as np
import random
from torch.utils.data import DataLoader, dataloader
from dataset import HemorrhageDatasetProvider
from model import HemorrhageDetector
from utils import get_argument_parser
from logger import Logger
from tqdm import tqdm

writer = SummaryWriter("runs")


def get_arguments():
    parser = get_argument_parser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    # no cuda mode is not supported
    args.no_cuda = False

    return args


def construct_arguments():
    args = get_arguments()

    # Prepare Logger
    logger = Logger(cuda=torch.cuda.is_available() and not args.no_cuda)
    args.logger = logger
    config = json.load(open(args.config_file, "r", encoding="utf-8"))
    args.config = config

    args.job_name = config["name"] if args.job_name is None else args.job_name
    print("Running Config File: ", args.job_name)
    # Setting the distributed variables
    print(f"Args = {args}")

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    args.saved_model_path = os.path.join(
        args.output_dir, "saved_models/", args.job_name
    )

    return args


def prepare_model_optimizer(args):
    args.local_rank = int(os.environ["LOCAL_RANK"])

    model = HemorrhageDetector(n_dim=1024)
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=parameters
    )

    return model_engine, optimizer


def run(args, model_engine, optimizer):
    config = args.config
    logger = args.logger

    dataset_provider = HemorrhageDatasetProvider(args)
    dataloaders = dataset_provider.get_dataloaders()

    criterion = nn.CrossEntropyLoss()

    for index in range(config["training"]["num_epochs"]):
        logger.info(f"Training epoch: {index + 1}")
        for op in dataloaders:

            if op == "train":
                model_engine.train()
            if op == "val":
                model_engine.eval()

            data_iterator = dataloaders[op]

            running_loss = 0.0
            op_series = 0

            for i, data in enumerate(tqdm(data_iterator)):
                series = data.series
                targets = data.targets
                labels = data.labels
                limits = data.limits

                series = series.to(model_engine.local_rank)
                targets = targets.to(model_engine.local_rank)

                outputs = model_engine(series, limits)
                loss = criterion(outputs, targets)
                print(list(zip(outputs, targets)))
                model_engine.backward(loss)
                model_engine.step()

                running_loss += loss.item() * outputs.size(0)
                op_series += outputs.size(0)

                if i % 100 == 99:
                    writer.add_scalar(
                        f"{op} loss", running_loss / 99, index * len(data_iterator) + i
                    )
                    print(running_loss / 99)
                    running_loss = 0.0

            epoch_loss = running_loss / op_series

            logger.info(f"{op} loss: {epoch_loss:.4f}")


args = construct_arguments()
model_engine, optimizer = prepare_model_optimizer(args)
args.train_micro_batch_size_per_gpu = model_engine.train_micro_batch_size_per_gpu()
run(args, model_engine, optimizer)

# data_dir = "/export/gaon1/data/jteneggi/data/rsna-intracranial-hemorrhage-detection"

# stages = ["train", "val"]
# datasets = {stage: SeriesDataset(data_dir=data_dir, op=stage) for stage in stages}
# dataloaders = {
#     "train": DataLoader(datasets["train"], batch_size=4, shuffle=True, num_workers=4),
#     "val": DataLoader(datasets["val"], batch_size=1, shuffle=False, num_workers=0),
# }
