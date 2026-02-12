from typing import Union,List, Optional
from pathlib import Path
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.ipc as ipc
import re, warnings, unicodedata
import shutil, os, random
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from .datasetSchema import schema

class TextNormalizer:
    """
    Normalize a transcription string by:
      - Lowercasing
      - Removing punctuation
      - Removing annotations in brackets: [], (), {}, <>
      - Removing diacritics (accents)
      - Collapsing multiple spaces
      - Trimming
    """
    @staticmethod
    def batch_normalize(txt) -> Union[str, List[str]]:
        if isinstance(txt, list):
            return [TextNormalizer.normalize(s) for s in txt]
        else:
            return TextNormalizer.normalize(txt)


    @staticmethod
    def normalize(text: str) -> str:
        if text is None:
            return ""

        # 1. Lowercase (uncomment if needed)
        # text = text.lower()

        # 2. Remove bracketed annotations like [noise], (laughs), {inaudible}, <annotation>
        text = re.sub(r"\[.*?\]", " ", text)
        text = re.sub(r"\(.*?\)", " ", text)
        text = re.sub(r"\{.*?\}", " ", text)
        text = re.sub(r"\<.*?\>", " ", text)

        # 3. Remove punctuation and symbols (keep letters, numbers, and spaces)
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        text = re.sub(r"[_]", " ", text)  # remove underscores explicitly if needed

        # 4. Remove diacritics (accents)
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

        # 5. Collapse multiple spaces and trim
        text = re.sub(r"\s+", " ", text).strip()

        return text

class DataHandler:
    """
    Handler for Whisper dataset preprocessing.
    Modified to work with multiprocessing by using instance methods.
    """
    
    def __init__(self, feature_extractor=None, tokenizer=None):
        """
        Initialize DataHandler with feature extractor and tokenizer.
        
        Args:
            feature_extractor: Whisper feature extractor
            tokenizer: Whisper tokenizer
        """
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
    
    def set_processors(self, FeatureExtractor, Tokenizer):
        """Set the feature extractor and tokenizer."""
        self.feature_extractor = FeatureExtractor
        self.tokenizer = Tokenizer
    
    def prepare_dataset(self, batch):
        """
        Prepare a single batch for training.
        
        Args:
            batch: Dictionary containing 'audio' and 'sentence' keys
            
        Returns:
            Dictionary with 'input_features' and 'labels'
        """
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = self.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # encode target text to label ids 
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch
    
    @staticmethod
    def view_table(table):
        df = table.to_pandas()
        for index, row in df.iterrows():
            print(f"Row {index}: {row.to_dict()}")
    
    @staticmethod
    def filter_by_duration(arrowFiles: List, threshold: float):
        # Reference schema: take from first shard
        print(f"Found {len(arrowFiles)} shards. Processing...")

        # extract the order of the columns in the shards to help in concatenation later
        ref_columns = pl.scan_ipc(str(arrowFiles[0])).columns
        
        for file in arrowFiles:
            print(f"Processing {file} ...")
            lf = pl.scan_ipc(str(file))  # Lazy load
            lf = lf.select(ref_columns)        # Re-order the columns
            
            # Filter out rows with duration >= threshold
            lf = lf.filter(pl.col("duration") >= threshold)
            
            lf.collect().write_ipc(file.replace(".arrow", "_filtered.arrow"))
            print(f"✅ Wrote filtered file → {file.replace('.arrow', '_filtered.arrow')}")

    @staticmethod
    def split_arrow_shards(input_dir: Path, train_out_dir, test_out_dir, split_col="split"):
        """
        Merge all Arrow shards into a single train and a single test shard,
        streaming shard by shard to avoid memory overflow.
        Handles shards with different schemas by aligning columns.
        Filters out rows with duration > 30.0.
        """
        input_dir = Path(input_dir)
        train_out_dir = Path(train_out_dir) 
        test_out_dir = Path(test_out_dir) 
        train_out_dir.mkdir(parents=True, exist_ok=True) 
        test_out_dir.mkdir(parents=True, exist_ok=True)

        train_out_file = train_out_dir / 'train.arrow'
        test_out_file = test_out_dir / 'test.arrow'

        # Remove existing output files if any
        if train_out_file.exists(): 
            train_out_file.unlink()
        if test_out_file.exists():  
            test_out_file.unlink()

        shard_files = list(input_dir.glob("*.arrow"))
        if not shard_files:
            raise FileNotFoundError(f"No .arrow files found in {input_dir}")

        # Reference schema: take from first shard
        print(f"Found {len(shard_files)} shards. Processing...")

        train_lfs = []
        test_lfs = []

        # extract the order of the columns in the shards to help in concatenation later
        ref_columns = pl.scan_ipc(str(shard_files[0])).columns

        for shard_path in shard_files:
            print(f"Processing {shard_path} ...")
            lf = pl.scan_ipc(str(shard_path))  # Lazy load
            lf = lf.select(ref_columns)        # Re-order the columns
            
            # Filter out rows with duration > 30.0
            lf = lf.filter(pl.col("duration") <= 30.0)
            
            train_lfs.append(lf.filter(pl.col(split_col) == "train"))
            test_lfs.append(lf.filter(pl.col(split_col) == "test"))

        # Vertically stack all train/test LazyFrames
        if train_lfs:
            train_all = pl.concat(train_lfs, how="vertical_relaxed")
            train_all.collect().write_ipc(train_out_file)
            print(f"✅ Wrote train file → {train_out_file}")

        if test_lfs:
            test_all = pl.concat(test_lfs, how="vertical_relaxed")
            test_all.collect().write_ipc(test_out_file)
            print(f"✅ Wrote test file → {test_out_file}")

        print("🎉 Done!")

    @staticmethod
    def read_arrow_head(file_path: str, n: int = 5, **kwargs) -> pa.Table:
        """
        Read the first n rows from an Arrow file, efficiently handling files larger than RAM.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        imp_columns = ['text', 'duration', 'split', 'OG_corpus']
        
        # Determine file format and use appropriate reader
        if file_path.endswith(('.arrow', '.feather')):
            return DataHandler._read_arrow_format_head(file_path, n, imp_columns, **kwargs)
        elif file_path.endswith('.parquet'):
            return DataHandler._read_parquet_head(file_path, n, imp_columns, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    @staticmethod
    def _read_arrow_format_head(file_path: str, n: int, columns: Optional[List[str]] = None, **kwargs) -> pa.Table:
        """Read head from Arrow/Feather format files with selected columns"""
        try:
            source = pa.memory_map(file_path)
            reader = pa.RecordBatchFileReader(source)
            
            # If columns specified, validate against schema
            if columns is not None:
                schema = reader.schema
                available_columns = schema.names
                valid_columns = [col for col in columns if col in available_columns]
                if not valid_columns:
                    valid_columns = available_columns
            else:
                valid_columns = None
            
            batches = []
            total_rows = 0
            
            for i in range(reader.num_record_batches):
                if total_rows >= n:
                    break
                    
                batch = reader.get_batch(i)
                
                if valid_columns is not None:
                    batch = batch.select(valid_columns)
                    
                batches.append(batch)
                total_rows += batch.num_rows
                
            if batches:
                table = pa.Table.from_batches(batches)
                return table.slice(0, min(n, table.num_rows))
            else:
                return pa.Table.from_batches([])
                
        except Exception as e:
            full_table = pa.ipc.open_file(file_path).read_all()
            result = full_table.slice(0, min(n, full_table.num_rows))
            
            if columns is not None:
                result = result.select(columns)
                
            return result

    @staticmethod
    def _read_parquet_head(file_path: str, n: int, **kwargs) -> pa.Table:
        """Read head from Parquet files using row group optimization"""
        return pq.read_table(file_path, **kwargs).slice(0, n)

    @staticmethod
    def read_arrow_head_streaming(file_path: str, n: int = 5, batch_size: int = 10000) -> pa.Table:
        """
        Read the first n rows using streaming approach for very large files.
        """
        if file_path.endswith('.parquet'):
            return DataHandler._read_parquet_head(file_path, n)
        
        batches = []
        total_rows = 0
        
        with pa.ipc.open_file(file_path) as reader:
            for i in range(reader.num_record_batches):
                if total_rows >= n:
                    break
                    
                batch = reader.get_batch(i)
                batches.append(batch)
                total_rows += batch.num_rows
        
        if batches:
            table = pa.Table.from_batches(batches)
            return table.slice(0, min(n, table.num_rows))
        else:
            return pa.Table.from_batches([])

"""----------HELPER FUNCTIONS----------"""
def process_data(audioPath: Path):
    txt_path = audioPath.with_suffix(".txt")
    if not txt_path.exists():
        warnings.warn(f"Skipping {audioPath} because {txt_path} not found.")
        return None
    try:
        audio_bytes = audioPath.read_bytes()
        sentence = txt_path.read_text(encoding="utf-8").strip()
        corpus_name = audioPath.parent.name
        info = sf.info(audioPath)
        duration = info.frames / info.samplerate
        
        split = "test" if "test" in str(audioPath).lower() else "train"
        
        return {
            "audio": {"bytes": audio_bytes, "path": str(audioPath)},
            "text": sentence,
            "duration": duration,
            "split": split,
            "OG_corpus": corpus_name,
            "utt_id": "None",
            "lang": "None",
            "task": "None",
            "translation_en": "None",
            "original_audio_id": "None",
            "original_audio_offset": 0.0,
                }
    except Exception as e:
        warnings.warn(f"Failed to process {audioPath}: {e}")
        return None
    
def parallel_record_generator(wav_files, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_data, wav): wav for wav in wav_files}
        for future in as_completed(futures):
            rec = future.result()
            if rec is not None:
                yield rec

def stream_WAV_to_shards(
    wav_files,
    out_dir="dataset_shards",
    rows_per_shard=10000,
    batch_size=512,
    max_workers=4
):
    #out_dir = Path(out_dir)
    #out_dir.mkdir(exist_ok=True)

    buffer = []
    shard_idx = 0
    row_count = 0
    sink = None
    writer = None

    for rec in parallel_record_generator(wav_files, max_workers=max_workers):
        buffer.append(rec)
        row_count += 1

        # flush batch
        if len(buffer) >= batch_size:
            table = pa.Table.from_pylist(buffer, schema=schema)
            if writer is None:  # open new shard
                shard_path = f"{out_dir}_shard{shard_idx}.arrow"
                sink = pa.OSFile(shard_path, "wb")
                writer = ipc.new_file(sink, table.schema)
            writer.write_table(table)
            buffer.clear()

        # close shard if limit reached
        if row_count >= rows_per_shard:
            if buffer:
                table = pa.Table.from_pylist(buffer, schema=schema)
                writer.write_table(table)
                buffer.clear()
            writer.close()
            sink.close()
            print(f"✅ Written shard_{shard_idx}.arrow with {row_count} rows")

            shard_idx += 1
            row_count = 0
            writer = None
            sink = None

    # flush final shard
    if buffer:
        rem = len(buffer)
        table = pa.Table.from_pylist(buffer, schema=schema)
        if writer is None:
            shard_path = f"{out_dir}_shard{shard_idx}.arrow"
            sink = pa.OSFile(shard_path, "wb")
            writer = ipc.new_file(sink, table.schema)
        writer.write_table(table)
        buffer.clear()

    if writer is not None:
        writer.close()
        sink.close()
        print(f"✅ Final shard_{shard_idx}.arrow with {rem} rows.")

def shard_parquet_to_arrow(file_path:str,shard_size:int,output_dir:str,shard_prefix:str):
    subset_dir = Path(output_dir) / shard_prefix
    subset_dir.mkdir(parents=True, exist_ok=True)

    i = 0
    total_rows = pq.ParquetFile(file_path).metadata.num_rows
    lf         = pl.scan_parquet(file_path)
    for offset in range(0, total_rows, shard_size):
        lf_chunk = lf.slice(offset, shard_size)
        try:
            #lf_chunk.sink_ipc(Path(f"{output_dir}/{shard_prefix}_shard{chunk_idx}.arrow"))
            arrow_table = lf_chunk.collect().to_arrow()
            ds.write_dataset(
                            data=arrow_table,
                            base_dir=subset_dir.as_posix(),  # folder where shards will be stored
                            format="ipc",           # or "arrow"
                            basename_template="shard-{i}.arrow",  # custom shard names
                            max_rows_per_group=10000
                        )
            os.replace(Path(output_dir) / shard_prefix / "shard-0.arrow", Path(output_dir) / shard_prefix / f"{shard_prefix}_shard{i}.arrow")
            print(f"✅ Saved rows to shard {i+1}.")
        except Exception as e:
            print("Caught exception:", e)
            import traceback
            traceback.print_exc()
            raise

        for file in subset_dir.iterdir():
            shutil.move(file, Path(output_dir) / file.name)
        i += 1
    # Remove the now-empty subset_dir
    os.rmdir(subset_dir)

def read_ipc_nommap(path: Path):
    """Read IPC/Arrow file without memory mapping."""
    with path.open("rb") as f:
        reader = ipc.open_file(f)  # disable mmap
        table = reader.read_all()
    return pl.from_arrow(table)

def update_split(subsets:list,base_path:Path,test_size:int=0.048):
        def update_shard_split(shard_path: Path, test_fraction: float = 0.048, seed: int = 42):
            # Step 1: Read without mmap
            df = read_ipc_nommap(shard_path)

            # Step 2: Compute train/test split
            n = df.height
            test_size = max(1, int(n * test_fraction))
            random.seed(seed)
            test_idx = set(random.sample(range(n), test_size))

            df = (
                df.with_row_index("idx")
                .with_columns(
                    pl.when(pl.col("idx").is_in(test_idx))
                    .then(pl.lit("test"))
                    .otherwise(pl.lit("train"))
                    .alias("split")
                )
                .drop("idx")
            )

            # Step 2b: If shard_path contains "YODAS", add OG_corpus column
            if "YODAS" in shard_path.name or "YODAS" in str(shard_path):
                df = df.with_columns(pl.lit("yodas_granary").alias("OG_corpus"))
                print(f"📌 Added OG_corpus column to {shard_path}")

            # Step 3: Write to temporary file
            tmp_path = shard_path.with_suffix(".tmp.arrow")
            df.write_ipc(tmp_path)

            # Step 4: Replace original safely
            os.replace(tmp_path, shard_path)
            print(f"✅ Updated {shard_path} with {test_size} test rows")
            
        test_size = test_size
        for name in subsets:
            #lf = load_subset_lazy(subset_name=name,base_dir='dataset')
            #df = assign_train_test(lf, test_fraction=0.048)
            #grouped = df.group_by("split").len()
            #print(grouped)
            base_path = Path(base_path)
            shard_files = sorted(base_path.glob(f"{name}_shard*.arrow"))
            if not shard_files:
                raise FileNotFoundError(f"No shards found for subset {name}")

            print(f"Processing {name}: {len(shard_files)} shards")
            for shard_path in shard_files:
                update_shard_split(shard_path, test_fraction=test_size, seed=42)
            print(f"✅ Completed updating all shards for {name}")

def make_test_train_split(dataset_dir:str,split_col:str="split"):
    """
    Merge all Arrow shards into a single train and a single test shard,
    streaming shard by shard to avoid memory overflow.
    Handles shards with different schemas by aligning columns.
    Filters out rows with duration > 30.0.
    """
    input_dir = Path(dataset_dir)
    train_out_dir = Path(input_dir.parent / "train") 
    test_out_dir = Path(input_dir.parent / "test") 
    train_out_dir.mkdir(parents=True, exist_ok=True) 
    test_out_dir.mkdir(parents=True, exist_ok=True)

    train_out_file = train_out_dir / 'train.arrow'
    test_out_file  = test_out_dir / 'test.arrow'

    # Remove existing output files if any
    if train_out_file.exists(): train_out_file.unlink()
    if test_out_file.exists():  test_out_file.unlink()

    shard_files = list(input_dir.glob("*.arrow"))
    if not shard_files:
        raise FileNotFoundError(f"No .arrow files found in {input_dir}")

    # Reference schema: take from first shard
    print(f"Found {len(shard_files)} shards. Processing...")

    train_lfs = []
    test_lfs = []

    # extract the order of the columns in the shards to help in concatination later
    ref_columns = pl.scan_ipc(str(shard_files[0])).columns

    for shard_path in shard_files:
        print(f"Processing {shard_path} ...")
        lf = pl.scan_ipc(str(shard_path))  # Lazy load
        lf = lf.select(ref_columns)        # Re-order the columns
        
        # Filter out rows with duration > 30.0
        lf = lf.filter(pl.col("duration") <= 30.0)
        
        train_lfs.append(lf.filter(pl.col(split_col) == "train"))
        test_lfs.append(lf.filter(pl.col(split_col) == "test"))

    # Vertically stack all train/test LazyFrames
    if train_lfs:
        train_all = pl.concat(train_lfs, how="vertical_relaxed")
        train_all.collect().write_ipc(train_out_file)
        print(f"✅ Wrote train file → {train_out_file}")

    if test_lfs:
        test_all = pl.concat(test_lfs, how="vertical_relaxed")
        test_all.collect().write_ipc(test_out_file)
        print(f"✅ Wrote test file → {test_out_file}")

    print("🎉 Done!")
#