import argparse
import sentencepiece as spm
import glob 

def main(args):
    data_dir = args.data_dir
    FILES = glob.glob(data_dir+'*')
    print(FILES)
    # we may consider adding --user_defined_symbols=INDENT,DEDENT,NEW_LINE
    
    spm.SentencePieceTrainer.train(
        '--input={} --vocab_size=1000 --model_prefix=sentencepiece.bpe '
        '--character_coverage=1.0 --model_type=bpe'.format(','.join(FILES))
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help='data directory')
    args = parser.parse_args()
    main(args)
