import pyarrow.dataset as ds
import torch, torchaudio
from torch.utils.data import get_worker_info,IterableDataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os, io, random

class ArrowAudioDataset(IterableDataset):
    def __init__(self, arrow_dir, tokeniser=None, pad=None, cache_dir=None, use_cache=False, transform=None, feature_extractor=None, tokenizer=None):
        """
        Args:
            arrow_dir: path to dataset directory or Arrow file
            cache_dir: optional directory for precomputed spectrograms
            use_cache: if True, save/load spectrograms from disk
            transform: optional transform to apply to spectrogram
            feature_extractor: Whisper feature extractor (passed directly for multiprocessing)
            tokenizer: Whisper tokenizer (passed directly for multiprocessing)
        """
        self.dataset = ds.dataset(arrow_dir, format="arrow")
        self.total_rows = sum(fragment.count_rows() for fragment in self.dataset.get_fragments())
        self.fragments = list(self.dataset.get_fragments())

        self.transform = transform
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.tokeniser = tokeniser  
        self.pad_trim = pad
        
        # Store feature extractor and tokenizer for multiprocessing compatibility
        self.feature_extractor = feature_extractor
        self.tokenizer_obj = tokenizer
        
        if self.use_cache:
            if not cache_dir:
                raise ValueError("cache_dir must be provided if use_cache=True")
            os.makedirs(cache_dir, exist_ok=True)
    
    def __len__(self):
        """Return the total number of rows in the dataset."""
        return self.total_rows
    
    def _process_row(self, row, idx, min_frames=10):
        """Decode and preprocess a single row."""
        # STEP-1 _____ load the audio bytes from the row
        try:
            audio_bytes = row["audio"][0]["bytes"]
            text = row["text"][0]
            label = text
        except Exception as e:
            print(f"[BAD ROW FORMAT] idx={idx} err={e}")
            return None

        cache_path = None
        if self.use_cache:
            cache_path = os.path.join(self.cache_dir, f"{idx}.pt")
            if os.path.exists(cache_path):
                spectrogram = torch.load(cache_path)
                # Skip too-short cached items
                if spectrogram.shape[-1] < min_frames:
                    print(f"[SKIPPED SHORT AUDIO] label='{label}' length={spectrogram.shape[-1]} frames")
                    return None
                return {
                    "input_ids": spectrogram, 
                    "labels": label,
                    "dec_input_ids": text,
                }

        # STEP-2 ____ Create the waveform from the bytes
        try:
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
            # Remove channel dimension
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
                sr = 16000
            waveform = waveform.squeeze(0)
            batch = {'audio': {'array': waveform, 'sampling_rate': sr}, 'sentence': text}
        except Exception as e:
            print(f"[TORCHAUDIO LOAD FAIL] idx={idx} label='{label}' error={e}")
            return None
        
        # STEP-3 ____ Apply the desired transform
        if self.transform:
            batch = self.transform(batch)

        if self.use_cache and cache_path:
            torch.save(batch, cache_path)

        return batch

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        for i, frag in enumerate(self.fragments):
            if i % num_workers != worker_id:
                continue

            table = frag.to_table()
            rows = table.to_batches()
            idx = 0
            for batch in rows:
                pdict = batch.to_pydict()
                num_rows = len(pdict["text"])
                for j in range(num_rows):
                    row = {k: [v[j]] for k, v in pdict.items()}
                    processed = self._process_row(row, idx)
                    if processed is not None:
                        yield processed
                    idx += 1

class ShuffleDataset(IterableDataset):
    """
    Shuffle buffer for IterableDataset.
    Note: this is not a true shuffle as it only takes in <buffer_size> number 
    of samples in order and then chooses randomly from it.
    """
    def __init__(self, dataset, buffer_size=8):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __len__(self):
        """Return the length of the underlying dataset."""
        if hasattr(self.dataset, 'total_rows') and self.dataset.total_rows is not None:
            return self.dataset.total_rows
        elif hasattr(self.dataset, '__len__'):
            return len(self.dataset)
        raise TypeError("Length unknown for IterableDataset")
 
    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            buf_size = self.buffer_size
            # Fill the buffer
            for i in range(buf_size):
                shufbuf.append(next(dataset_iter))
        except StopIteration:
            # If dataset is smaller than buffer_size
            buf_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, buf_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            # Empty the remaining buffer
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    feature_extractor: Any
    tokenizer: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch