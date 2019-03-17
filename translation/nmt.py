import argparse
import random
import sys
from model import *
from train import *
from eval import *
from utils import *

def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # Network
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Network depth.")
    parser.add_argument("--num_encoder_layers", type=int, default=2,
                        help="Encoder depth, equal to num_layers if None.")
    parser.add_argument("--num_decoder_layers", type=int, default=2,
                        help="Decoder depth, equal to num_layers if None.")
    parser.add_argument("--input_size", type=int, default=128, help="Input size.")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size.")

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--num_train_steps", type=int, default=10000,
                        help="Num steps to train.")

    # Data
    parser.add_argument("--src", type=str, default=None,
                        help="Source suffix, e.g., en.")
    parser.add_argument("--tgt", type=str, default=None,
                        help="Target suffix, e.g., de.")
    parser.add_argument("--out_dir", type=str, default='model',
                        help="Store log/model files.")

    # Sequence lengths
    parser.add_argument("--src_max_len", type=int, default=50,
                        help="Max length of src sequences during training.")
    parser.add_argument("--tgt_max_len", type=int, default=50,
                        help="Max length of tgt sequences during training.")

    # Default setting
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate (not keep_prob)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                        help="Clip gradients to this norm.")

    # Evaluation
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint file to load a model for inference.")

def main():
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    flags, unparsed = nmt_parser.parse_known_args()

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    input_lang, output_lang, pairs = prepareData(flags.src, flags.tgt, flags.src_max_len, flags.tgt_max_len)
    pairs = trimRareWords(input_lang, output_lang, pairs, min_count=3)

    if flags.ckpt:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(flags.ckpt)
        # If loading a model trained on GPU to CPU
        #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        input_embedding_sd = checkpoint['input_embedding']
        output_embedding_sd = checkpoint['output_embedding']
        input_lang.__dict__ = checkpoint['input_lang_dict']
        output_lang.__dict__ = checkpoint['output_lang_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    input_embedding = nn.Embedding(input_lang.num_words, flags.hidden_size)
    output_embedding = nn.Embedding(output_lang.num_words, flags.hidden_size)
    if flags.ckpt:
        input_embedding.load_state_dict(input_embedding_sd)
        output_embedding.load_state_dict(output_embedding_sd)
        # Initialize encoder & decoder models

    attn_model = 'dot'
    encoder = EncoderRNN(flags.hidden_size, input_embedding, flags.num_encoder_layers, flags.dropout)
    decoder = LuongAttnDecoderRNN(attn_model, output_embedding, flags.hidden_size, output_lang.num_words, flags.num_decoder_layers, flags.dropout)
    if flags.ckpt:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=flags.learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=flags.learning_rate)
    if flags.ckpt:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # Run training iterations
    print("Starting Training!")
    print(flags.ckpt)
    trainIters(flags, input_lang, output_lang, pairs, encoder, decoder, encoder_optimizer,
               decoder_optimizer, input_embedding, output_embedding, device)

    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()
    evaluateInput(encoder, decoder, input_lang, output_lang, flags.tgt_max_len, device)

if __name__ == "__main__":
    main()
