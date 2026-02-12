from pathlib import Path
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from utils import ArrowAudioDataset,ShuffleDataset, DataHandler, DataCollatorSpeechSeq2SeqWithPadding
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import sys


if __name__ == '__main__':
    dataset_dir = sys.argv[1]  # "Path/to/arrows"
    lang        = sys.argv[2]  # "Italian"

    data_dir = Path(dataset_dir)
    train_dir= data_dir / 'train'
    test_dir = data_dir / 'test'


    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    t  = WhisperTokenizer.from_pretrained("openai/whisper-base",language=lang,task="transcribe")
    dh = DataHandler()
    dh.set_processors(FeatureExtractor=fe,Tokenizer=t)

    train_set= ArrowAudioDataset(arrow_dir=train_dir,transform=dh.prepare_dataset)
    train_shu= ShuffleDataset(dataset=train_set)
    test_set= ArrowAudioDataset(arrow_dir=test_dir,transform=dh.prepare_dataset)
    test_shu= ShuffleDataset(dataset=test_set)

    # Checking the formating with just one item
    
    iterator = iter(train_set)
    first_item = next(iterator)
    input_str = first_item['sentence']
    labels    = first_item['labels']
    decoded_with_special = t.decode(labels,skip_special_tokens=False)
    decoded_str= t.decode(labels,skip_special_tokens=True)

    print(f"Lables coming out of tokeniser: {labels}")
    print(f"Input:                 {input_str}")
    print(f"Decoded w/ special:    {decoded_with_special}")
    print(f"Decoded w/out special: {decoded_str}")
    print(f"Are equal:             {input_str == decoded_str}")


    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    model.generation_config.language = "italian" #somehow i stupidly left it at hindi in the first Run
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        feature_extractor=fe,
        tokenizer=t,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    import evaluate
    #from datasets import load_metric
    metric = evaluate.load("wer")
    #metric = load_metric("wer")
    def compute_metrics(pred):
        import numpy as np
        preds = pred.predictions
        labels = pred.label_ids

        # Some HF models output tuple (logits, ...)
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 ignored index
        labels = np.where(labels != -100, labels, t.pad_token_id)

        # Decode
        pred_str = t.batch_decode(preds, skip_special_tokens=True)
        label_str = t.batch_decode(labels, skip_special_tokens=True)

        # FILTER OUT EMPTY REFERENCES
        filtered_preds = []
        filtered_labels = []

        for p, l in zip(pred_str, label_str):
            if l is None:
                continue
            l_clean = l.strip()
            if len(l_clean) == 0:
                continue  # skip empty labels completely

            filtered_preds.append(p.strip())
            filtered_labels.append(l_clean)

        # SAFETY CHECK
        if len(filtered_labels) == 0:
            # Avoid crashing JIWER
            return {"wer": 0.0}

        # ✔️ USE ONLY FILTERED LISTS
        wer_value = metric.compute(
            predictions=filtered_preds,
            references=filtered_labels
        )

        return {"wer": float(wer_value)}
    
    import wandb
    wandb.login()

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-base-it",  # change to a repo name of your choice
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=10000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_shu,
        eval_dataset=test_shu,
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=fe,
    )

    trainer.train()


    