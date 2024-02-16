from argparse import ArgumentParser


if __name__ == '__main__':

    parser = ArgumentParser(description='Evaluate faitfulness')
    
    parser.add_argument('--input_data_dir', 
        help='Output directory in which explanations will be stored', 
        default='~/data/legal', 
        required=False,
        type=str)
    

    import pandas as pd
    
    args = parser.parse_args()

    input_data_dir = args.input_data_dir
    a = pd.read_csv(f'{input_data_dir}/test_data5doc_sentences.csv')
    print(a)