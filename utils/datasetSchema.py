import pyarrow as pa

schema = pa.schema([
    ("audio", pa.struct([
        ("bytes", pa.binary()),
        ("path", pa.string())
    ])),
    ("text", pa.string()),
    ("duration", pa.float64()),
    ("split", pa.string()),
    ("OG_corpus", pa.string()),
    # Extra fields for Parquet compatibility
    ("utt_id", pa.string()),
    ("lang", pa.string()),
    ("task", pa.string()),
    ("translation_en", pa.string()),
    ("original_audio_id", pa.string()),
    ("original_audio_offset", pa.float64())
])