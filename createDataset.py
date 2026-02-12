import os
from pathlib import Path
import argparse
from utils import stream_WAV_to_shards, shard_parquet_to_arrow,update_split,make_test_train_split

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="Task runner")

    subparsers = parser.add_subparsers(dest="task", required=True)

    # ---- Task 1 ----
    task1_parser = subparsers.add_parser("1", help="Make arrow shards from .wav and .txt files.")
    task1_parser.add_argument("--corpus_dir",
        type=str,
        default=str("corpus"),
        help="Root corpus directory")
    task1_parser.add_argument("--dataset_dir",
        type=str,
        default=str("dataset"),
        help="Root output directory for the final dataset shards")
    task1_parser.add_argument("--names",
        nargs="+",                 # ← this is the key
        default=[
            'APASCI','FLEURS','LABLITA','LIBRISPEECH',
            'SPEECON','VOXFORGE','VOXPOPULI','CLIPS_LETTO'
        ],
        help="List of dataset names")

    # ---- Task 2 ----
    task2_parser = subparsers.add_parser("2", help="make arrow shards from single .parquet file.")
    task2_parser.add_argument("--file_path", type=str, required=True)
    task2_parser.add_argument("--output_path", type=str, required=True)
    task2_parser.add_argument("--shard_name", type=str, required=True)
    task2_parser.add_argument("--shard_size", type=int, default=20000)

    # ---- Task 3 ----
    task3_parser = subparsers.add_parser("3", help="add info about train/test split in the arrow files.")
    task3_parser.add_argument("--dataset_names", nargs="+",help="List of dataset names that are un-partioned.",required=True)
    task3_parser.add_argument("--dataset_dir", type=str, required=True)
    task3_parser.add_argument("--test_size",type=float,default=0.048)

    # ---- Task 4 ----
    task4_parser = subparsers.add_parser("4", help="create a train/test split from the arrow files.")
    task4_parser.add_argument("--dataset_path",help="pass dataset directory to create test-train split.",required=True)
    task4_parser.add_argument("--split_col", type=str,help="specify the column name to use for finding the split type." ,required=True)

    args = parser.parse_args()

    if args.task == "1":  #wav 2 arrow shards
        print(f"Extracting audio and text files from {args.corpus_dir} for the following datasets = {args.names}\nfinal output directory = {args.dataset_dir}")

        main_dir    = args.corpus_dir
        dataset_dir = args.dataset_dir
        names       = args.names

        for name in names:
            p = Path(main_dir) / name
            print("Checking:", p)
            print("Exists:", p.exists())
            print("Is dir:", p.is_dir())
            wav_files = list(Path(os.path.join(main_dir,name)).rglob("*.wav"))
        
            stream_WAV_to_shards(
            wav_files,
            out_dir=str(os.path.join(dataset_dir,f"{name}")),
            rows_per_shard=10000,
            batch_size=10000,
            max_workers=8  # tune depending on CPU + disk speed
            )

    elif args.task == "2": # parquet to arrow shards
        file_path   = args.file_path
        output_dir  = args.output_path
        prefix      = args.shard_name
        shard_size  = args.shard_size
        print(f"converting {file_path} with batch size {shard_size} with prefix : {prefix}\noutput path : {output_dir}")

        shard_parquet_to_arrow(file_path=file_path,output_dir=output_dir,shard_prefix='YODAS',shard_size=20000)

    elif args.task == "3": # split shards
        dataset_names = args.dataset_names
        dataset_dir   = args.dataset_dir
        test_size     = args.test_size
        print(f"Partitioning {dataset_names} with a test size of {test_size}\noutput dir {dataset_dir}")
        update_split(subsets=dataset_names,base_path=dataset_dir,test_size=test_size)

    elif args.task == "4":
        dataset_dir   = args.dataset_path
        split_col     = args.split_col
        print(f"creating test-train files from {dataset_dir} using the column : {split_col}")
        make_test_train_split(dataset_dir=dataset_dir,split_col=split_col)

    # ===========================================
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus_dir",
        type=str,
        default=str("corpus"),
        help="Root corpus directory"
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str("dataset"),
        help="Root output directory for the final dataset shards"
    )

    parser.add_argument(
        "--names",
        nargs="+",                 # ← this is the key
        default=[
            'APASCI','FLEURS','LABLITA','LIBRISPEECH',
            'SPEECON','VOXFORGE','VOXPOPULI','CLIPS_LETTO'
        ],
        help="List of dataset names"
    )

    args = parser.parse_args()

    main_dir    = args.corpus_dir
    dataset_dir = args.dataset_dir
    names       = args.names

    #main_dir = Path("corpus")
    #names = ['APASCI','FLEURS','LABLITA','LIBRISPEECH','SPEECON','VOXFORGE','VOXPOPULI','CLIPS_LETTO']
    
    for name in names:
        p = Path(main_dir) / name
        print("Checking:", p)
        print("Exists:", p.exists())
        print("Is dir:", p.is_dir())
        wav_files = list(Path(os.path.join(main_dir,name)).rglob("*.wav"))
        
        stream_WAV_to_shards(
        wav_files,
        out_dir=str(os.path.join(dataset_dir,f"{name}")),
        rows_per_shard=10000,
        batch_size=10000,
        max_workers=8  # tune depending on CPU + disk speed
        )
    
    #
    
    shard_parquet_to_arrow(file_path='yodas_granary_IT.parquet',output_dir='dataset',shard_prefix='YODAS',shard_size=20000)
    """
    # update the splits on unsplit subsets.
    '''
    split_updateSubsets(subsets=['APASCI','CLIPS','VOXFORGE','YODAS'],base_path=Path("dataset"))
    '''

    # create train-test sub directory
    # IMPORTANT NOTE: This does not delete the original shards and thus doubles the size occupied on disk.
    # IMPORTANT NOTE: This next step also removes all the rows with audio longer than 30 seconds.
    '''
    split_arrow_shards(input_dir='dataset',train_out_dir='dataset/train',test_out_dir='dataset/test') 
    '''

    '''
    arrow_files = [
    "dataset/train/train.arrow",
    "dataset/test/test.arrow",
    ]
    list(map(lambda f: filter_arrow_by_duration(f, max_value=30.0), arrow_files))
    '''
    