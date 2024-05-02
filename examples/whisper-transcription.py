import whisper
import intel_npu_acceleration_library
import torch
import os, io, time
import numpy as np
from scipy.io import wavfile

def audio_to_float(audio):
    """
    convert audio signal to floating point format
    """
    return audio.astype(np.float32) / np.iinfo(audio.dtype).max

def resample(audio, src_sample_rate, dst_sample_rate):
    """
    Resample audio to specific sample rate

    Parameters:
      audio: input audio signal
      src_sample_rate: source audio sample rate
      dst_sample_rate: destination audio sample rate
    Returns:
      resampled_audio: input audio signal resampled with dst_sample_rate
    """
    if src_sample_rate == dst_sample_rate:
        return audio
    duration = audio.shape[0] / src_sample_rate
    resampled_data = np.zeros(shape=(int(duration * dst_sample_rate)), dtype=np.float32)
    x_old = np.linspace(0, duration, audio.shape[0], dtype=np.float32)
    x_new = np.linspace(0, duration, resampled_data.shape[0], dtype=np.float32)
    resampled_audio = np.interp(x_new, x_old, audio)
    return resampled_audio.astype(np.float32)

def get_audio(wavefile):
    sample_rate, audio = wavfile.read(io.BytesIO(open(wavefile, 'rb').read()))
    audio = audio_to_float(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # The model expects mono-channel audio with a 16000 Hz sample rate, represented in floating point range. When the
    # audio from the input video does not meet these requirements, we will need to apply preprocessing.
    resampled_audio = resample(audio, sample_rate, 16000)

    # Get the duration in seconds
    duration_seconds = len(resampled_audio) / 1000.0
    
    # Print the duration
    print(f"The duration of the audio file is {duration_seconds:.2f} seconds.")

    return resampled_audio, duration_seconds

def calculate_wer(reference, hypothesis):
	ref_words = reference.split()
	hyp_words = hypothesis.split()
	# Counting the number of substitutions, deletions, and insertions
	substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
	deletions = len(ref_words) - len(hyp_words)
	insertions = len(hyp_words) - len(ref_words)
	# Total number of words in the reference text
	total_words = len(ref_words)
	# Calculating the Word Error Rate (WER)
	wer = (substitutions + deletions + insertions) / total_words
	return wer

if __name__=='__main__':    
    support_models = ["tiny", "base", "medium"]
    
    model = whisper.load_model(support_models[2], "cpu")
    
    print("Compile model for the NPU")
    optimized_model = intel_npu_acceleration_library.compile(model, dtype=torch.int8)

    # Ready to inference
    prefix_path = 'cv-valid-test-partial/'
    label_file = 'cv-valid-test-partial/cv-valid-test.csv'
    
    inference_data = {}
    label_data = open(label_file)
    label_data = label_data.readlines()
    for line in label_data[1:]:
        elements = line.split(',')
        inference_data[elements[0]] = elements[1]

    for filename, label in inference_data.items():
        resampled_audio, duration = get_audio(prefix_path+filename)
        # Record the start time
        start_time = time.time()
        transcription = optimized_model.transcribe(resampled_audio, task="transcribe")
        end_time = time.time()
        # Calculate the execution time
        execution_time = end_time - start_time
        print(transcription['text'])
        print('WER(word error rate): ' + str(round(calculate_wer(label, transcription['text']),2)))
        # Print the execution time
        print(f"Execution time: {execution_time:.4f} seconds")
 


