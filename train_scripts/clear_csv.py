import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--csv_file", required=True)
    args = parser.parse_args()

    images_dir = args.images_dir
    csv_path = args.csv_file

    df = pd.read_csv(csv_path, encoding='utf-8')
    
    for index, row in df.iterrows():
        print(row['image'])
        try:
            f = open(row['image'])
        except IOError:
            df.drop(index, inplace=True)
   
    df.to_csv(csv_path, encoding='utf-8', index=False)
