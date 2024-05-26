from transformers import pipeline

transcriber = pipeline(task="automatic-speech-recognition")
print(transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"))

transcriber = pipeline(model="openai/whisper-large-v2")
print(transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"))

print(transcriber("https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/ted_60.wav"))