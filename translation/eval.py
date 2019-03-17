import torch.nn as nn
from utils import *

def evaluate(encoder, decoder, input_lang, output_lang, sentence, max_length, device):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(input_lang, sentence)]
    # Create lengths tensor
    input_length = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_seq = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_seq = input_seq.to(device)
    input_length = input_length.to(device)
    # Forward input through encoder model
    encoder_outputs, encoder_hidden = encoder(input_seq, input_length)
    # Prepare encoder's final hidden layer to be first hidden input to the decoder
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    # Initialize decoder input with SOS_token
    decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * 1
    # Initialize tensors to append decoded words to
    all_tokens = torch.zeros([0], device=device, dtype=torch.long)
    # Iteratively decode one word token at a time
    for _ in range(max_length):
        # Forward pass through decoder
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        # Obtain most likely word token and its softmax score
        decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
        #token = output_lang.index2word[decoder_input.item()]
        if decoder_input == 0 or decoder_input == 2: break
        all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
        # Prepare current token to be next decoder input (add a dimension)
        decoder_input = torch.unsqueeze(decoder_input, 0)
    # indexes -> words
    decoded_words = [output_lang.index2word[token.item()] for token in all_tokens]
    return decoded_words


def evaluateInput(encoder, decoder, input_lang, output_lang, tgt_max_len, device):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, input_lang, output_lang, input_sentence, tgt_max_len, device)
            print('<', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")
