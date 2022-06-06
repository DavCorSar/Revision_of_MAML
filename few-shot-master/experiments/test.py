import torch
from torch.utils.data import DataLoader
from few_shot.eval import evaluate
from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.core import NShotTaskSampler

args = your_args
model = ModelClass()
model.load_state_dict('path/to/model/weights.pt')
dataloader = DataLoader(
    OmniglotDataset(),
    batch_sampler=NShotTaskSampler(evaluation, args.eval_batches, n=args.n, k=args.k, q=args.q,
                                   num_tasks=args.meta_batch_size),
    num_workers=8
)
prepare_batch = your_prepare_batch_function
metrics = evaluate(model, dataloader, your_prepare_batch_function, metrics=['accuracy'])
