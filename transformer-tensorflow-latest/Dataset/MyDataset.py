import numpy as np

def generate_random_data(total_sequences, max_input_length, max_target_length, input_vocab_size, target_vocab_size):
    """
    Generate random tokenized data for testing purposes.

    Args:
        total_sequences (int): Total number of sequences to generate.
        max_input_length (int): Maximum length of input sequences.
        max_target_length (int): Maximum length of target sequences.
        input_vocab_size (int): Vocabulary size for input sequences.
        target_vocab_size (int): Vocabulary size for target sequences.

    Returns:
        tuple: Four Numpy arrays:
            - input_seq (total_sequences, max_input_length): Random input sequences.
            - target_seq (total_sequences, max_target_length): Random target sequences.
            - input_padding_mask (total_sequences, max_input_length): Mask for input sequences.
            - target_padding_mask (total_sequences, max_target_length): Mask for target sequences.
    """
    # Generate random integers for input (context) and target sequences
    input_seq = np.random.randint(1, input_vocab_size, size=(total_sequences, max_input_length))
    target_seq = np.random.randint(1, target_vocab_size, size=(total_sequences, max_target_length))

    # Padding mask for input sequence (1 for padding, 0 for real tokens)
    input_padding_mask = np.where(input_seq == 0, 1, 0)  # Assuming '0' is padding token
    
    # Padding mask for target sequence (same concept for target padding mask)
    target_padding_mask = np.where(target_seq == 0, 1, 0)

    return input_seq, target_seq, input_padding_mask, target_padding_mask