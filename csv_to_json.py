import pandas as pd
import os
import jsonlines

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")

def create_dir(output):
  if not os.path.exists(output):
    os.makedirs(output)


def export_json_files(output_dir, filename, df, direction):

  to_be_saved = []
  src_data = df['source_lang'].values
  tgt_data = df['target_lang'].values
  src_lang, tgt_lang = direction.split('-')
  N_sent = df.shape[0]

  for s in range(N_sent):
    text_string = {"translation": {src_lang:src_data[s], tgt_lang:tgt_data[s]}}
    to_be_saved.append(text_string)

  with jsonlines.open(os.path.join(output_dir, filename), 'w') as writer:
    writer.write_all(to_be_saved)


def combine_texts_lafand(args, input_path, output_path, direction, n_sent=50000, header=None):

  # output data
  output_dir = output_path
  create_dir(output_dir)

  src_lang, tgt_lang = direction.split('-')

  if args["train"]:

    df_train = pd.read_csv(os.path.join(input_path, 'train.tsv'), sep='\t', header=header) #[For CSV, use sep=',']
    sc_train, tg_train = df_train.iloc[:n_sent, df_train.columns.get_loc(src_lang)].values, df_train.iloc[:n_sent, df_train.columns.get_loc(tgt_lang)].values

    df_train_sctg = pd.DataFrame(sc_train, columns=['source_lang'])
    df_train_sctg['target_lang'] = tg_train

    export_json_files(output_dir, 'train.json', df_train_sctg, direction=direction)

  if args["dev"]:
    
    df_dev = pd.read_csv(os.path.join(input_path, 'dev.tsv'), sep='\t', header=header)
    sc_dev, tg_dev = df_dev.iloc[:, df_dev.columns.get_loc(src_lang)].values, df_dev.iloc[:, df_dev.columns.get_loc(tgt_lang)].values
    
    df_dev_sctg = pd.DataFrame(sc_dev, columns=['source_lang'])
    df_dev_sctg['target_lang'] = tg_dev

    export_json_files(output_dir, 'dev.json', df_dev_sctg, direction=direction)
  
  if args["test"]:
    
    df_test = pd.read_csv(os.path.join(input_path, 'test.tsv'), sep='\t', header=header)
    sc_test, tg_test = df_test.iloc[:, df_test.columns.get_loc(src_lang)].values, df_test.iloc[:, df_test.columns.get_loc(tgt_lang)].values

    df_test_sctg = pd.DataFrame(sc_test, columns=['source_lang'])
    df_test_sctg['target_lang'] = tg_test

    export_json_files(output_dir, 'test.json', df_test_sctg, direction=direction)


if __name__ == "__main__":
    
  parser = ArgumentParser(description="Converting parallel data in tsv to json",formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument("input", help="input directory")
  parser.add_argument("output", help="output directory")
  parser.add_argument("direction", help="translation direction. example: hau-yor for Hausa to Yoruba")
  parser.add_argument("-n", "--n_sent", type=int, help="no. of sentences", default=50000)
  parser.add_argument("--has_header", action="store_true", help="indicates if files have header")
  parser.add_argument("--train", action="store_true", help="create data for training")
  parser.add_argument("--test", action="store_true", help="create data for testing")
  parser.add_argument("--dev", action="store_true", help="create data for validation")
  args = vars(parser.parse_args())

  input_path = args["input"]
  output_path = args["output"]
  trans_direction = args["direction"]
  
  header = None
  if args["has_header"]:
    header = 0
  
  combine_texts_lafand(args, input_path, output_path, trans_direction, 50000, header)
  print('Conversion done!!!')
