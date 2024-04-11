from collections import Counter


def add_token(vocab, token):
    """
    Add a token to the vocab. Check if the token is already in the vocab, if not, add it to the vocab.

    Args:
        vocab (dict): The vocab dictionary
        token (str): The token to add to the vocab

    Returns:
        int: The index of the token in the vocab
    """
    if token not in vocab['word2idx']:
        vocab['word2idx'][token] = vocab['count']
        vocab['idx2word'][vocab['count']] = token
        vocab['count'] += 1
        return vocab['count'] - 1
    else:
        return vocab['word2idx'][token]


def add_tokens(vocab, tokens):
    """
    Add a list of tokens to the vocab. Check if the token is already in the vocab, if not, add it to the vocab.

    Args:
        vocab (dict): A dictionary of vocabularies.
        tokens (list): A list of tokens to be added to the vocab.

    Returns:
        list: A list of indices of the tokens in the vocab.
    """
    indices = []
    for token in tokens:
        index = add_token(vocab, token)
        indices.append(index)
    return indices


def initialize_vocabulary():
    """
    Initialize the vocabulary with the following tokens:
    <UNK>: Unknown token
    <PAD>: Padding token
    <EOS>: End of sentence token

    There are two different dictionaries: word2idx and idx2word

    Args:
        None

    Returns:
        vocab (dict): A dictionary of vocabularies.
    """

    # Define the vocab
    vocab = {}

    # Define the special tokens
    special_tokens = ['<UNK>', '<PAD>', '<EOS>']

    # Define the word2idx dictionary
    vocab['word2idx'] = {}

    # Define the idx2word dictionary
    vocab['idx2word'] = {}

    # Define the count
    vocab['count'] = 0

    # Add the special tokens to the vocab
    for token in special_tokens:
        add_token(vocab, token)

    return vocab


def lookup_token(vocab, token):
    """
    Look up a token in the vocab.

    Args:
        vocab (dict): A dictionary of vocabularies.
        token (str): A token to be looked up.

    Returns:
        int: The index of the token in the vocab, or the index of the <UNK> token if the token is not in the vocab.
    """
    return vocab['word2idx'].get(token, vocab['word2idx']['<UNK>'])


def lookup_index(vocab, index):
    """
    Look up an index in the vocab.

    Args:
        vocab (dict): A dictionary of vocabularies.
        index (int): An index to be looked up.

    Returns:
        str: The token corresponding to the index in the vocab, or the <UNK> token if the index is not in the vocab.
    """
    return vocab['idx2word'].get(index, '<UNK>')


def vocabulary_from_tokens(tokens, cutoff=25):
    """
    Build the vocab from a dataframe.

    Args:
        tokens: a list of tokens to be added to the vocab
        cutoff: the minimum frequency of the token to be added to the vocab

    Returns:
        vocab: a dictionary of vocabularies
    """
    # Initialize the vocab
    vocab = initialize_vocabulary()

    # Loop through all the tokens and count the frequency of each token
    word_counter = Counter(tokens)

    # Add the token to the vocab if the frequency is greater than cutoff
    for word, count in word_counter.items():
        if count > cutoff:
            add_token(vocab, word)

    return vocab
