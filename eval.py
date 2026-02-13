import torch
import torchaudio
import re
import warnings
import polars as pl
from pathlib import Path
import argparse
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils import TextNormalizer


# ---------------------------------------------------------
# 0. Helper Functions
# ---------------------------------------------------------

def make_path_sentence_dict(df: pl.DataFrame) -> dict:
    """Convert Polars dataframe to {path: sentence} dict."""
    if not {"path", "sentence"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'path' and 'sentence'")
    return dict(zip(df["path"].to_list(), df["sentence"].to_list()))

def extract_input_files(sh_file_path):
    """Extract wav filenames from a shell script."""
    pattern = r"doasr_home\.py\s+/home/docker/([A-Za-z0-9_\-]+\.wav)"
    files = []
    with open(sh_file_path, "r") as f:
        for line in f:
            m = re.search(pattern, line)
            if m:
                files.append(m.group(1))
    return files

def load_table(fPath: str | Path, columns: list[str] | None = None) -> pl.DataFrame:
    fPath = Path(fPath)
    if not fPath.exists():
        raise FileNotFoundError(f"{fPath} does not exist.")

    suffix = fPath.suffix.lower()

    if suffix == ".xlsx":
        df = pl.read_excel(fPath, columns=columns)
    elif suffix == ".csv":
        df = pl.read_csv(fPath, columns=columns)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. Only .xlsx and .csv are supported.")
    return df

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. Parse Arguments
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Inference mode for benchmarking.")
    # Required
    parser.add_argument("model_path", type=str, help="Path to the base model checkpoint") 
    parser.add_argument("lang",type=str,help="Pass the language code for model, Ex; It for Italian.")
    parser.add_argument("refText_path", type=str, help="Path to table containing audio paths & transciptions. Accepts both .xlsx and .csv")
    parser.add_argument("output_path", type=str, help="Path to directory for saving the outputs.")     
    # Optional flag
    parser.add_argument("--use_finetuned",action="store_true",help="Use fine-tuned weights")
    # Optional argument (only required if --use_finetuned is set)
    parser.add_argument("--finetuned_weights",type=str,default=None,help="Path to fine-tuned weights (required if --use_finetuned is set)")


    args = parser.parse_args()
    if args.use_finetuned and args.finetuned_weights is None:
        parser.error("--finetuned_weights must be provided when --use_finetuned is set.")

    model_path = args.model_path
    lang       = args.lang
    ckpt_path  = args.finetuned_weights  #"whisper-base-it/checkpoint-6000"
    output_dir = args.output_path

    # ---------------------------------------------------------
    # 2. Load Model 
    # ---------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load processor
    #   |   distill large-v3 : bofenghuang/whisper-large-v3-distil-it-v0.2
    #   |   normal base      : openai/whisper-base
    processor = WhisperProcessor.from_pretrained(model_path,language=lang,task="transcribe")
    if args.use_finetuned:
        model = WhisperForConditionalGeneration.from_pretrained(ckpt_path)
    else:
        # load base model architecture 
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # 3. Load table + Build Reference Dict
    # ---------------------------------------------------------
    test_names = extract_input_files("asr_NVIDIA_FastConformer_Hybrid_Large (1).sh")
    xlPath = "SELECTED/Selection_of_validated_speech.xlsx"
    df     = load_table(fPath=xlPath,columns=["path","sentence","up_votes",])
    df     = df.filter(pl.col("up_votes") >= 5)
    ref_dict = make_path_sentence_dict(df)
    normalizer = TextNormalizer()

    # ---------------------------------------------------------
    # 4. Preload Audio into RAM (Big Speedup)
    # ---------------------------------------------------------

    audio_cache = {}
    print("Preloading audio...")
    snr = 'clean'
    for name in test_names:
        full_path = f"SELECTED_NOISY/validated_upvotes_5_delta_{snr}/{name}"
        if not Path(full_path).exists():
            warnings.warn(f"⚠️ File not found: {full_path}")
            continue
        
        audio, sr = torchaudio.load(full_path)
        audio_cache[name] = (audio, sr)

    # ---------------------------------------------------------
    # 5. Evaluation Loop (Optimized)
    # ---------------------------------------------------------
    hyp_out = Path(f"{output_dir}hyp.txt")
    ref_out = Path(f"{output_dir}ref.txt")

    with open(hyp_out, "w", encoding="utf-8") as hyp_file, \
        open(ref_out, "w", encoding="utf-8") as ref_file:

        for name in test_names:
            if name not in audio_cache:
                continue
            
            audio, sr = audio_cache[name]

            # resample to 16k if needed
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)

            # processor handles batching and padding
            inputs = processor(
                audio.squeeze(),
                sampling_rate=16000,
                return_tensors="pt"
            ).to(device)

            # inference_mode is faster than no_grad
            with torch.inference_mode():
                predicted_ids = model.generate(
                    inputs["input_features"],
                    language='it',
                    num_beams=1,     # fast greedy decoding
                    do_sample=False
                )

            transcription = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            hyp = normalizer.normalize(transcription)
            ref = normalizer.normalize(ref_dict[name.removesuffix(".wav") + ".mp3"])

            utt_id = name.removesuffix(".wav")

            hyp_file.write(f"{hyp} ({utt_id})\n")
            ref_file.write(f"{ref} ({utt_id})\n")

    print("✔ Evaluation completed.")




