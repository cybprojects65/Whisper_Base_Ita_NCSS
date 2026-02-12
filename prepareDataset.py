''' PART 1 -- Normalise the text in arrow files.'''
''' PART 2 -- Split the arrow files in train and test set.'''

from pathlib import Path
import sys
from tqdm import tqdm
import traceback
import pyarrow.ipc as ipc
import pyarrow as pa
import argparse

from utils import TextNormalizer, DataHandler

def normalize_text(arrow_files):
    tn = TextNormalizer()
    for arrow_file in tqdm(arrow_files, desc="Processing Arrow files", unit="file"):
        try:
            # Read the Arrow file
            table = pa.ipc.RecordBatchFileReader(arrow_file).read_all()

            # Skip files without 'text' column
            if 'text' not in table.column_names:
                continue

            # Apply preprocessing
            text_array = table['text']
            processed_text = pa.array([tn.batch_normalize(x.as_py()) for x in text_array])  #x is not a native Python string, it’s a PyArrow scalar (type: pyarrow.lib.StringScalar).

            # Replace the 'text' column
            new_table = table.set_column(table.schema.get_field_index('text'), 'text', processed_text)

            # Save back to the same file using a temporary file
            tmp_file = arrow_file.with_suffix(".tmp.arrow")
            with pa.OSFile(str(tmp_file), "wb") as sink:
                with ipc.new_file(sink, new_table.schema) as writer:
                    writer.write_table(new_table)

            # Atomic replace
            tmp_file.replace(arrow_file)

        except Exception as e:
            print(f"\tError processing {arrow_file.name}: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--filter-by-duration",action="store_true",help="Enable duration filtering")

    args = parser.parse_args()
    dataset_dir = args.data_dir
    filter_by_duration = args.filter_by_duration
    #dataset_dir = sys.argv[1]  # "Path/to/arrowsFolder"
    #filter_by_duration = sys.argv[2]  # False

    all_files_dir = Path(f"{dataset_dir}/all")
    all_files = list(all_files_dir.glob("*.arrow"))
    dh = DataHandler()
    """
    #STEP 0 --> Check out how the files are right now
    myTable = dh.read_arrow_head(str(all_files[0]),n=10)
    dh.view_table(myTable)
    """
    #STEP 1 -->
    normalize_text(all_files)
    
    #STEP 2 -->
    dh.split_arrow_shards(input_dir=all_files_dir,train_out_dir=f"{dataset_dir}/train",test_out_dir=f"{dataset_dir}/test")
    
    #STEP 2.1 -- check the resultant data (sanity check of sorts)
    if filter_by_duration:
        limit=0.5
        dh.filter_by_duration(arrowFiles=[f'{dataset_dir}/train/train.arrow'],threshold=limit)

        formated_table = dh.read_arrow_head(f'{dataset_dir}/train/train_filtered.arrow',n=10)
        dh.view_table(formated_table)
    