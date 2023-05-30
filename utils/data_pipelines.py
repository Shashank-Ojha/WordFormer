# Utils to read and parse text data
import torch

def read_file(filename):
  """Reads a file and returns the text as a single string."""
  with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()
  return text


def get_vocab(text):
  """Returns a sorted list of unique tokens in the text and the the number
  of unique tokens."""
  vocab = sorted(list(set(text)))
  vocab_size = len(vocab)
  return vocab, vocab_size

def train_validation_split(data, train_percent):
  """Splits the data into training and validation sets."""
  n = int(train_percent*len(data)) # first train_percent will be train, rest val
  train_data = data[:n]
  val_data = data[n:]
  return train_data, val_data


def get_batch(data, batch_size, block_size):
  """Returns batches of data.

  The returned batch (x, y) is of size (batch_size, block_size).
  This actually contains batch_size * block_size supervised examples. This is
  because X[i, 0:j] is a context containing upto block_size tokens. The
  corresponding Y[i, j] is the target token that follows the context.

  data: tensor with shape (datasize,)
  """
  # Randomly samples the data for the beginning of a continuous sequence of
  # block_size+1 tokens.
  # Shape (batch_size, )
  ix = torch.randint(len(data) - (block_size + 1), (batch_size,))

  # Shape (batch_size, block_size)
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x, y