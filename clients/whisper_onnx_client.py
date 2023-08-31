# Test script to run the pipeline once it has been loaded in triton
import tritonclient.http as httpclient
import numpy as np
import torchaudio
import argparse


if __name__ == "__main__":
    # Setting up client
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Load and preprocess audio data
    audio, sample_rate = torchaudio.load("gettysburg.wav")
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        audio = resampler(audio)
        audio = audio.squeeze(0)

    # Create input tensor
    input_tensor = httpclient.InferInput("audio", audio.shape, datatype="FP32")
    input_tensor.set_data_from_numpy(audio.numpy())

    # Query the server
    response = client.infer(model_name="whisper_base_onnx", inputs=[input_tensor])

    # Get the output tensor
    output_tensor = response.as_numpy("transcription")

    print("Transcription:", output_tensor[0].decode("utf-8"))
