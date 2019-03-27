import os
import argparse
import random
import time

from utils import Processor, TrainDataset, TestDataset

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


def get_tp_fp_fn(logits, labels):
  assert labels.shape[1] == 1
  labels = labels.squeeze()
  predictions = np.argmax(logits, axis=1)
  labels, predictions = labels.astype(int), predictions.astype(int)
  tp = np.sum(np.logical_and(predictions == 1, labels == 1))
  fp = np.sum(np.logical_and(predictions == 1, labels == 0))
  fn = np.sum(np.logical_and(predictions == 0, labels == 1))
  return tp, fp, fn


def compute_metrics(tp, fp, fn):
  precision = tp / (tp + fp + np.finfo(float).eps)
  recall = tp / (tp + fn + np.finfo(float).eps)
  f1 = 2 * precision * recall / (precision + recall + np.finfo(float).eps)
  return precision, recall, f1


def warmup_linear(x, warmup=0.002):
  if x < warmup:
    return x / warmup
  return 1.0 - x


def main():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument("--type",
                      type=str,
                      required=True,
                      help="The type other than Drug in the relation")

  # Some parameters
  parser.add_argument("--bert_model",
                      default="bert-base-uncased",
                      type=str,
                      help="Bert pre-trained model selected in the list: bert-base-uncased, "
                           "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
  parser.add_argument("--output_dir",
                      default="../out/",
                      type=str,
                      help="The output directory where the model predictions and checkpoints will be written.")
  parser.add_argument("--strategy",
                      default="edge",
                      type=str,
                      help="The strategy to select negative examples for training")
  parser.add_argument("--max_seq_length",
                      default=512,
                      type=int,
                      help="The maximum total input sequence length after WordPiece tokenization. \n"
                           "Sequences longer than this will be truncated, and sequences shorter \n"
                           "than this will be padded.")
  parser.add_argument("--val_split",
                      type=float,
                      default=0.2,
                      help="The fraction of spliting validation set from the whole training set")
  parser.add_argument("--edge_percentage",
                      default=80,
                      type=int,
                      help="Percentage of selecting adjacent negative examples")
  parser.add_argument("--mask_concepts",
                      default=False,
                      action='store_true',
                      help="Whether to mask two concepts.")

  parser.add_argument("--do_train",
                      default=False,
                      action='store_true',
                      help="Whether to run training.")
  parser.add_argument("--do_test",
                      default=False,
                      action='store_true',
                      help="Whether to run test on the test set.")
  parser.add_argument("--test_final",
                      default=False,
                      action='store_true',
                      help="Whether to test the model of final training step. If not specified,\n will test the model with best f1 score in valdiation set")
  parser.add_argument("--do_lower_case",
                      default=True,
                      action='store_false',
                      help="Set this flag if you are using an uncased model.")
  parser.add_argument("--train_batch_size",
                      default=16,
                      type=int,
                      help="Total batch size for training.")
  parser.add_argument("--eval_batch_size",
                      default=8,
                      type=int,
                      help="Total batch size for eval.")
  parser.add_argument("--learning_rate",
                      default=2e-5,
                      type=float,
                      help="The initial learning rate for Adam.")
  parser.add_argument("--num_train_epochs",
                      default=10.0,
                      type=float,
                      help="Total number of training epochs to perform.")
  parser.add_argument("--warmup_proportion",
                      default=0.1,
                      type=float,
                      help="Proportion of training to perform linear learning rate warmup for. "
                           "E.g., 0.1 = 10%% of training.")
  parser.add_argument("--no_cuda",
                      default=False,
                      action='store_true',
                      help="Whether not to use CUDA when available")
  parser.add_argument("--local_rank",
                      type=int,
                      default=-1,
                      help="local_rank for distributed training on gpus")
  parser.add_argument('--seed',
                      type=int,
                      default=42,
                      help="random seed for initialization")
  parser.add_argument('--gradient_accumulation_steps',
                      type=int,
                      default=1,
                      help="Number of updates steps to accumulate before performing a backward/update pass.")
  parser.add_argument('--fp16',
                      default=False,
                      action='store_true',
                      help="Whether to use 16-bit float precision instead of 32-bit")
  parser.add_argument('--loss_scale',
                      type=float, default=0,
                      help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                           "0 (default value): dynamic loss scaling.\n"
                           "Positive power of 2: static loss scaling value.\n")

  args = parser.parse_args()

  best_model_file = os.path.join(args.output_dir, "best_model.bin")
  final_model_file = os.path.join(args.output_dir, "final_model.bin")

  if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
  else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
  print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
      device, n_gpu, bool(args.local_rank != -1), args.fp16))

  if args.gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
        args.gradient_accumulation_steps))

  if args.train_batch_size % 2 != 0:
    raise ValueError("Invalid train_batch_size parameter: {}, should be an even number".format(
        args.train_batch_size))

  args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

  # Set random seeds
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

  if not args.do_train and not args.do_test:
    raise ValueError("At least one of `do_train` or `do_test` must be True.")

  # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
  #   raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
  os.makedirs(args.output_dir, exist_ok=True)

  tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

  processor = Processor(tokenizer, args.max_seq_length, args.type, args.val_split)
  label_list = processor.get_labels()
  num_labels = len(label_list)

  # For do_train is True, the model train on training set and evaluate on validation set at
  # the end of each epoch, and save the best model
  if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # # Prepare model
    # model = BertForSequenceClassification.from_pretrained(
    #     args.bert_model,
    #     cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
    #     num_labels=num_labels)

    # if args.fp16:
    #   model.half()
    # model.to(device)
    # if args.local_rank != -1:
    #   try:
    #     from apex.parallel import DistributedDataParallel as DDP
    #   except ImportError:
    #     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    #   model = DDP(model)
    # elif n_gpu > 1:
    #   model = torch.nn.DataParallel(model)

    # global_step = 0
    # The examples_list is [pos_train, neg_candidate_train, neg_rest_train, pos_val, neg_val]
    examples_list, mask_list, neg_indices_train, val_false_negtive = processor.get_examples(args.mask_concepts)
    print("Number of pos_train:{} pos_val:{} neg_val:{}"
          .format(examples_list[0].shape[0], examples_list[3].shape[0], examples_list[4].shape[0]))
    num_train_steps = int(
        2 * len(examples_list[0]) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
      t_total = t_total // torch.distributed.get_world_size()

    if args.fp16:
      try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
      except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

      optimizer = FusedAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            bias_correction=False,
                            max_grad_norm=1.0)
      if args.loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
      else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
      optimizer = BertAdam(optimizer_grouped_parameters,
                           lr=args.learning_rate,
                           warmup=args.warmup_proportion,
                           t_total=t_total)

    # Prepare train set
    if args.strategy == 'edge':
      train_data = TrainDataset(
          pos_train=examples_list[0],
          neg_candidate_train=examples_list[1],
          neg_rest_train=examples_list[2],
          pos_train_mask=mask_list[0],
          neg_candidate_train_mask=mask_list[1],
          neg_rest_train_mask=mask_list[2],
          neg_indices_train=neg_indices_train,
          edge_percentage=args.edge_percentage)
    elif args.strategy == 'random':
      train_data = TrainDataset(
          pos_train=examples_list[0],
          neg_candidate_train=examples_list[1],
          neg_rest_train=examples_list[2],
          pos_train_mask=mask_list[0],
          neg_candidate_train_mask=mask_list[1],
          neg_rest_train_mask=mask_list[2])

    if args.local_rank == -1:
      train_sampler = RandomSampler(train_data)
    else:
      train_sampler = DistributedSampler(train_data)
    train_sampler = BatchSampler(train_sampler, batch_size=int(args.train_batch_size / 2), drop_last=False)
    train_dataloader = DataLoader(train_data, batch_sampler=train_sampler, num_workers=0)

    # Prepare validation set
    eval_data = TestDataset(
        torch.tensor(examples_list[3], dtype=torch.long),
        torch.tensor(examples_list[4], dtype=torch.long),
        torch.tensor(mask_list[3], dtype=torch.long),
        torch.tensor(mask_list[4], dtype=torch.long))

    eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, num_workers=0)

    # Start training
    print('Start training...')
    best_f1 = 0
    for epoch in range(int(args.num_train_epochs)):
      # Record total loss for training set and validation set
      tr_loss, nb_tr_steps = 0, 0
      # Train for one epoch, and evaluate later
      model.train()
      for step, (pos_train, neg_train, pos_mask, neg_mask) in enumerate(train_dataloader):

        # Put tensors to devices
        input_ids = torch.cat([pos_train, neg_train]).to(device)
        input_mask = torch.cat([pos_mask, neg_mask]).to(device)
        segment_ids = torch.zeros_like(input_mask).to(device)
        label_ids = torch.tensor([1 for _ in range(pos_train.size(0))] + [0 for _ in range(neg_train.size(0))]).to(device)

        loss = model(input_ids, segment_ids, input_mask, label_ids)

        if n_gpu > 1:
          loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
          loss = loss / args.gradient_accumulation_steps

        if args.fp16:
          optimizer.backward(loss)
        else:
          loss.backward()

        tr_loss += loss.item()
        nb_tr_steps += 1
        # learning rate decay
        if (step + 1) % args.gradient_accumulation_steps == 0:
          # modify learning rate with special warm up BERT uses
          lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
          for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_step
          optimizer.step()
          optimizer.zero_grad()
          global_step += 1
      print("Epoch {} train loss={}, ".format(epoch, round(tr_loss / nb_tr_steps, 3)), end='')

      # Record tp, fp, fn for validation set
      val_tp, val_fp, val_fn, precision, recall, f1 = [0 for _ in range(6)]

      eval_loss, nb_eval_steps = 0, 0
      # Evaluate model on validation set
      model.eval()
      for input_ids, input_mask, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = torch.zeros_like(input_mask).to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
          tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
          logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.cpu().numpy()

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        tp, fp, fn = get_tp_fp_fn(logits, label_ids)
        val_tp += tp
        val_fp += fp
        val_fn += fn
      precision, recall, f1 = compute_metrics(val_tp, val_fp, val_fn + val_false_negtive)
      print("validation loss={}  precision={}  recall={}  f1={}"
            .format(round(eval_loss / nb_eval_steps, 3), round(precision, 3), round(recall, 3), round(f1, 3)))
      print("validation tp:{} fp:{} fn:{}".format(val_tp, val_fp, val_fn + val_false_negtive))
      # save a trained model if f1 is bigger than best_f1
      if f1 > best_f1:
        best_f1 = f1
        model_to_save = model.module if hasattr(model, 'module') else model

        torch.save(model_to_save.state_dict(), best_model_file)

    # save final model
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), final_model_file)

  # For test mode when do_test is True
  if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Prepare test data
    pos_test, neg_test, pos_test_mask, neg_test_mask, test_false_negative = processor.get_test_examples(args.mask_concepts)

    test_data = TestDataset(
        torch.tensor(pos_test, dtype=torch.long),
        torch.tensor(neg_test, dtype=torch.long),
        torch.tensor(pos_test_mask, dtype=torch.long),
        torch.tensor(neg_test_mask, dtype=torch.long))
    test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size, num_workers=0)

    # Load a trained model that have fine-tuned
    if args.test_final:
      model_state_dict = torch.load(final_model_file)
    else:
      model_state_dict = torch.load(best_model_file)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict)
    model.to(device)

    # Record tp, fp, fn for test set
    test_tp, test_fp, test_fn, precision, recall, f1 = [0 for _ in range(6)]

    # Evaluate model on test set
    model.eval()
    for input_ids, input_mask, label_ids in test_dataloader:
      input_ids = input_ids.to(device)
      input_mask = input_mask.to(device)
      segment_ids = torch.zeros_like(input_mask).to(device)
      label_ids = label_ids.to(device)

      with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask)

      logits = logits.detach().cpu().numpy()
      label_ids = label_ids.cpu().numpy()

      tp, fp, fn = get_tp_fp_fn(logits, label_ids)
      test_tp += tp
      test_fp += fp
      test_fn += fn
    precision, recall, f1 = compute_metrics(test_tp, test_fp, test_fn + test_false_negative)
    print("Test precision={}  recall={}  f1={}"
          .format(round(precision, 3), round(recall, 3), round(f1, 3)))
    print("Test tp:{} fp:{} fn:{}".format(test_tp, test_fp, test_fn + test_false_negative))


if __name__ == '__main__':
  start = time.time()
  main()
  period = int((time.time() - start) / 60)
  print("program run time: {} mins".format(period))
