import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def _extract_concepts(file, non_drug_type):
  """Extract drug and non-drug concepts from annotation files.
  Args:
    file: file to extract
    non_drug_type: a string, the type of the non-drug concept, for example, reason, frequency
  Return:
    non_drug_dict: a dict of non_drug concepts, the keys are concept indices like "T1", the values are tuples of (start offset, end offset)
    drug_dict: a dict of drug concepts
    relations: a list of tuples of (non_drug index, drug index), for example, ("T1", "T3")
  """
  non_drug_dict = {}
  drug_dict = {}
  relations = []

  with open(file, "r") as f:
    for line in f:
      segments = line.split()
      # for concepts, annotations start with "T"
      if line[0] == 'T':
        if segments[1] == 'Drug':
          # some concepts cross lines, the annotation uses two tuples with ';' to seperate them
          i = 0
          while ';' in segments[3 + i]:
            i += 1
          drug_dict[segments[0]] = (int(segments[2]), int(segments[3 + i]))

        elif segments[1] == non_drug_type:
          i = 0
          while ';' in segments[3 + i]:
            i += 1
          non_drug_dict[segments[0]] = (int(segments[2]), int(segments[3 + i]))

      elif line[0] == 'R':
        if segments[1] == non_drug_type + '-Drug':
          relations.append((segments[2][5:], segments[3][5:]))

  return non_drug_dict, drug_dict, relations


def _get_examples_for_single_file(non_drug_dict, drug_dict, relations, file, max_tokens, mask_concepts):
  """Get positive examples, candidate examples, and rest_examples"""
  non_drug_list = sorted(non_drug_dict.items(), key=lambda t: t[1][0])
  drug_list = sorted(drug_dict.items(), key=lambda t: t[1][0])

  # get positive examples
  pos_tuples = []
  for relation in relations:
    try:
      non_drug_tuple = non_drug_dict[relation[0]]
      drug_tuple = drug_dict[relation[1]]
      pos_tuples.append((non_drug_tuple, drug_tuple))
    except:
      pass

  non_drug_index_list = [non_drug[0] for non_drug in non_drug_list]
  drug_index_list = [drug[0] for drug in drug_list]

  # get candidate negative examples, which are examples are close to the positive example
  candidate_tuples = []
  candidate_indices = []
  for i, relation in enumerate(relations):
    candidate_non_drugs = []
    candidate_drugs = []
    # get index of each positive examples
    try:
      non_drug_index = non_drug_index_list.index(relation[0])
      drug_index = drug_index_list.index(relation[1])
    except:
      continue

    if non_drug_index >= 1:
      candidate_non_drugs.append(non_drug_list[non_drug_index - 1][1])
    if non_drug_index < len(non_drug_list) - 1:
      candidate_non_drugs.append(non_drug_list[non_drug_index + 1][1])

    if drug_index >= 1:
      candidate_drugs.append(drug_list[drug_index - 1][1])
    if drug_index < len(drug_list) - 1:
      candidate_drugs.append(drug_list[drug_index + 1][1])

    for non_drug in candidate_non_drugs:
      candidate_tuple = (non_drug, drug_list[drug_index][1])
      if candidate_tuple not in pos_tuples and candidate_tuple not in candidate_tuples:
        candidate_tuples.append(candidate_tuple)
        candidate_indices.append(i)

    for drug in candidate_drugs:
      candidate_tuple = (non_drug_list[non_drug_index][1], drug)
      if candidate_tuple not in pos_tuples and candidate_tuple not in candidate_tuples:
        candidate_tuples.append(candidate_tuple)
        candidate_indices.append(i)

  # get the rest of the negative examples, which are examples are not close to the positive examples:
  rest_tuples = []
  for non_drug in non_drug_dict.values():
    for drug in drug_dict.values():
      rest_tuple = (non_drug, drug)
      if rest_tuple not in pos_tuples and rest_tuple not in candidate_tuples:
        rest_tuples.append(rest_tuple)

  # assert bool(set(pos_tuples) & set(candidate_tuples)) ^ bool(set(pos_tuples) & set(rest_tuples)) ^ bool(set(candidate_tuples) & set(rest_tuples)) == False

  # get real examples (strings)
  pos_examples = []
  candidate_examples = []
  rest_examples = []

  # get positive examples
  false_negtives = 0
  for pos_tuple in pos_tuples:
    pos_example = _extract_single_string(file, pos_tuple[0], pos_tuple[1], max_tokens, mask_concepts)
    if pos_example:
      pos_examples.append(pos_example)
    else:
      false_negtives += 1

  # get candidate negative examples
  false_positives = 0
  keep_indices = [i for i in range(len(candidate_indices))]
  for i, candidate_tuple in enumerate(candidate_tuples):
    candidate_example = _extract_single_string(file, candidate_tuple[0], candidate_tuple[1], max_tokens, mask_concepts)
    if candidate_example:
      candidate_examples.append(candidate_example)
    else:
      keep_indices.remove(i)
      false_positives += 1
  origin_len = len(candidate_indices)
  candidate_indices = np.take(candidate_indices, keep_indices)
  assert false_positives + len(candidate_indices) == origin_len

  # get the rest of negative examples
  for rest_tuple in rest_tuples:
    rest_example = _extract_single_string(file, rest_tuple[0], rest_tuple[1], max_tokens, mask_concepts)
    if rest_example:
      rest_examples.append(rest_example)

  return pos_examples, candidate_examples, rest_examples, candidate_indices, false_negtives


def _extract_single_string(file, non_drug, drug, max_tokens, mask_concepts=False):
  """Extract a single string
  Args:
    non_drug: a non-drug tuple of (start offset, end offset)
    drug: a drug tuple of (start offset, end offset)
    max_tokens: only extract string if the number of tokens is less than max_tokens,
  Return:
    string: the extracted string
  """
  if non_drug[0] < drug[0]:
    concept1 = non_drug
    concept2 = drug
    # span = (non_drug[0], drug[1])
  else:
    # span = (drug[0], non_drug[1])
    concept1 = drug
    concept2 = non_drug

  with open(file, "r") as f:
    doc = f.read()
    string = doc[concept1[0]: concept2[1]]
    mid_string = string[concept1[1] - concept1[0]: concept2[0] - concept1[0]]
    if mask_concepts:
      result = "src " + mid_string + " tgt"
    else:
      con1_string = string[: concept1[1] - concept1[0]]
      con2_string = string[concept2[0] - concept2[1]:]
      result = "srcstart " + con1_string + ' srcend ' + mid_string + ' tgtstart ' + con2_string + " tgtend"
    if len(result.split()) > max_tokens - 1:
      return ""
    else:
      return result


def get_examples(non_drug_type, val_split, max_tokens, mask_concepts):
  """Get examples from all training data
  args:
    non_drug_type: str, the concept type in a relation other than Drug
    val_split: float, proportion of validation set
    max_tokens: int, the maximum sequence length, which depends on non_drug_type
    mask_concepts: bool, whether to mask the concepts
  """
  pos_train, candidate_train, rest_train, candidate_indices = [], [], [], []
  pos_val, neg_val = [], []
  val_false_negatives = 0

  train_path = 'dataset/track2-training-data'
  for fn in os.listdir(train_path):
    if fn[-4:] == '.ann':
      continue
    else:
      txt_fn = fn
      ann_fn = fn[:-4] + '.ann'
    to_set = 'train'
    if random.random() < val_split:
      to_set = 'val'
    non_drug_dict, drug_dict, relations = _extract_concepts(os.path.join(train_path, ann_fn), non_drug_type)
    pos_example, candidate_example, rest_example, candidate_index, false_negtive = _get_examples_for_single_file(non_drug_dict, drug_dict, relations, os.path.join(train_path, txt_fn), max_tokens, mask_concepts)
    if to_set == 'train':
      candidate_train.extend(candidate_example)
      candidate_index = len(pos_train) + candidate_index
      candidate_indices.extend(candidate_index.tolist())
      pos_train.extend(pos_example)
      rest_train.extend(rest_example)
    else:
      pos_val.extend(pos_example)
      neg_val.extend(candidate_example + rest_example)
      val_false_negatives += false_negtive
  print('Number of examples in training set, pos_train:{} , neg_candidate_train:{} , neg_train:{}'.
        format(len(pos_train), len(candidate_train), len(candidate_train) + len(rest_train)))

  # import statistics
  # lens = [len(seq.split()) for seq in pos_train]
  # print(f"mean: {round(statistics.mean(lens),3)} , sd: {round(statistics.stdev(lens),3)} , median: {statistics.median(lens)} , max:{max(lens)}")
  # lens = [len(seq.split()) for seq in candidate_train]
  # print(f"mean: {round(statistics.mean(lens),3)} , sd: {round(statistics.stdev(lens),3)} , median: {statistics.median(lens)} , max:{max(lens)}")
  # lens = [len(seq.split()) for seq in candidate_train + rest_train]
  # print(f"mean: {round(statistics.mean(lens),3)} , sd: {round(statistics.stdev(lens),3)} , median: {statistics.median(lens)} , max:{max(lens)}")

  # found = len(set(candidate_indices))
  # print(f'found: {found} , not_found: {len(pos_train) - found}')

  # print(candidate_indices[:50])
  # max_index, max_len = 0, 0
  # for i, pos in enumerate(pos_train):
  #   if len(pos.split()) > max_len:
  #     max_len = len(pos.split())
  #     max_index = i
  # index = [j for j, x in enumerate(candidate_indices) if x == max_index]

  # print(pos_train[i])
  # print([candidate_train[j] for j in index])

  # import pickle
  # lens = [len(seq.split()) for seq in candidate_train + rest_train]
  # save_name = 'out/' + non_drug_type
  # with open(save_name, 'wb') as f:
  #   pickle.dump(lens, f)
  # exit()

  return pos_train, candidate_train, rest_train, candidate_indices, pos_val, neg_val, val_false_negatives


def get_test_examples(non_drug_type, max_tokens, mask_concepts):
  """Get examples from all test data"""
  pos_test, neg_test = [], []
  test_false_negatives = 0

  test_path = 'dataset/track2-test-data'
  for fn in os.listdir(test_path):
    if fn[-4:] == '.ann':
      continue
    else:
      txt_fn = fn
      ann_fn = fn[:-4] + '.ann'

    non_drug_dict, drug_dict, relations = _extract_concepts(os.path.join(test_path, ann_fn), non_drug_type)
    pos_example, candidate_example, rest_example, _, false_negtive = _get_examples_for_single_file(non_drug_dict, drug_dict, relations, os.path.join(test_path, txt_fn), max_tokens, mask_concepts)

    pos_test.extend(pos_example)
    neg_test.extend(candidate_example + rest_example)
    test_false_negatives += false_negtive

  print('Number of positive test example: {} , negative test examples: {}'.format(len(pos_test), len(neg_test)))
  exit()
  return pos_test, neg_test, test_false_negatives


class Processor(object):
  def __init__(self, tokenizer, max_seq_length, non_drug_type, val_split):
    self.tokenizer = tokenizer
    self.max_seq_length = max_seq_length
    self.non_drug_type = non_drug_type
    self.val_split = val_split

  def _convert_single_example(self, example):
    """convert a single example from a string to a list of strings"""
    tokens = self.tokenizer.tokenize(example)
    # Account for [CLS] with "- 1"
    if len(tokens) > self.max_seq_length - 1:
      tokens = tokens[0:(self.max_seq_length - 1)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens.insert(0, "[CLS]")
    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 for _, _ in enumerate(input_ids)]

    # Zero-pad up to the sequence length.
    padding_len = self.max_seq_length - len(input_ids)
    input_ids.extend([0 for _ in range(padding_len)])
    input_mask.extend([0 for _ in range(padding_len)])

    assert len(input_ids) == self.max_seq_length, "{} {}".format(len(input_ids), self.max_seq_length)
    assert len(input_mask) == self.max_seq_length, "{} {}".format(len(input_mask), self.max_seq_length)

    return input_ids, input_mask

  def _get_ids_and_mask(self, examples_list, mask_list):
    for i, examples in enumerate(examples_list):
      mask = []
      for j, example in enumerate(examples):
        input_ids, input_mask = self._convert_single_example(example)
        examples[j] = input_ids
        mask.append(input_mask)

      # convert examples and mask to numpy array
      examples = np.asarray(examples)
      mask = np.asarray(mask)

      # update list
      examples_list[i] = examples
      mask_list.append(mask)

    return examples_list, mask_list

  def get_examples(self, mask_concepts):
    pos_train, neg_candidate_train, neg_rest_train, neg_indices_train, pos_val, neg_val, val_fn = \
        get_examples(self.non_drug_type, self.val_split, self.max_seq_length, mask_concepts)

    examples_list = [pos_train, neg_candidate_train, neg_rest_train, pos_val, neg_val]
    mask_list = []
    # segment_ids_list = []

    examples_list, mask_list = self._get_ids_and_mask(examples_list, mask_list)

    return examples_list, mask_list, np.asarray(neg_indices_train), val_fn

  def get_test_examples(self, mask_concepts):
    pos_test, neg_test, test_fn = get_test_examples(self.non_drug_type, self.max_seq_length, mask_concepts)
    examples_list = [pos_test, neg_test]
    mask_list = []
    examples_list, mask_list = self._get_ids_and_mask(examples_list, mask_list)
    return examples_list[0], examples_list[1], mask_list[0], mask_list[1], test_fn

  def get_labels(self):
    return ['related', 'not_related']


class TrainDataset(Dataset):
  def __init__(self, **kwargs):
    self.pos_train = kwargs["pos_train"]
    self.neg_candidate_train = kwargs["neg_candidate_train"]
    self.neg_rest_train = kwargs["neg_rest_train"]
    self.pos_train_mask = kwargs["pos_train_mask"]
    self.neg_candidate_train_mask = kwargs["neg_candidate_train_mask"]
    self.neg_rest_train_mask = kwargs["neg_rest_train_mask"]
    self.strategy = 'random'
    if "neg_indices_train" in kwargs:
      self.neg_indices_train = kwargs["neg_indices_train"]
      self.strategy = 'edge'
      self.edge_percentage = kwargs["edge_percentage"]

  def __getitem__(self, idx):
    # pick negative example given idx
    if self.strategy == 'edge':
      indices = np.where(self.neg_indices_train == idx)[0]
      # pick neg example from candidate set if indices exist
      if len(indices) != 0 and random.randint(1, 100) <= self.edge_percentage:
        index = np.random.choice(indices, 1).squeeze()
        neg_example = self.neg_candidate_train[index]
        neg_mask = self.neg_candidate_train_mask[index]
      # if indices don't exist, pick neg example from the rest set
      else:
        index = np.random.choice(self.neg_rest_train.shape[0], 1).squeeze()
        neg_example = self.neg_rest_train[index]
        neg_mask = self.neg_rest_train_mask[index]

    elif self.strategy == 'random':
      num_of_negs = self.neg_candidate_train.shape[0] + self.neg_rest_train.shape[0]
      index = random.randint(0, num_of_negs - 1)
      _index = index - self.neg_candidate_train.shape[0]
      if _index < 0:
        neg_example = self.neg_candidate_train[index]
        neg_mask = self.neg_candidate_train_mask[index]
      else:
        neg_example = self.neg_rest_train[_index]
        neg_mask = self.neg_rest_train_mask[_index]

    # Get pos example
    pos_example = self.pos_train[idx]
    pos_mask = self.pos_train_mask[idx]
    assert pos_example.shape == pos_mask.shape == neg_example.shape == neg_mask.shape, "{} {} {} {}".format(pos_example.shape, pos_mask.shape, neg_example.shape, neg_mask.shape)
    assert pos_example.ndim == 1, "rank of pos_example is {}".format(pos_example.ndim)

    # Convert numpy arrays to tensors
    pos_example = torch.tensor(pos_example, dtype=torch.long)
    neg_example = torch.tensor(neg_example, dtype=torch.long)
    pos_mask = torch.tensor(pos_mask, dtype=torch.long)
    neg_mask = torch.tensor(neg_mask, dtype=torch.long)

    return (pos_example, neg_example, pos_mask, neg_mask)

  def __len__(self):
    return self.pos_train.shape[0]


class TestDataset(Dataset):
  """Prepare dataset for validation set and test set"""

  def __init__(self, *tensors):
    self.pos = tensors[0]
    self.neg = tensors[1]
    self.pos_mask = tensors[2]
    self.neg_mask = tensors[3]

  def __getitem__(self, idx):
    _idx = idx - self.pos.size(0)
    if _idx < 0:
      return (self.pos[idx], self.pos_mask[idx], torch.tensor([1], dtype=torch.long))
    else:
      return (self.neg[_idx], self.neg_mask[_idx], torch.tensor([0], dtype=torch.long))

  def __len__(self):
    return self.pos.size(0) + self.neg.size(0)
