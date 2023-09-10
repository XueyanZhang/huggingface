from transformers import pipeline

speech = ["https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
          "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/i-know-kung-fu.mp3"]
generator = pipeline(model="openai/whisper-large", device=0)
results = generator(speech)
for r in results:
    print(r)