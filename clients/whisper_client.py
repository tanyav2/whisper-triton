# Test script to run the pipeline once it has been loaded in triton
import tritonclient.http as httpclient
import numpy as np
import torchaudio


if __name__ == "__main__":
    # Setting up client
    client = httpclient.InferenceServerClient(url="localhost:9000")

    # Load and preprocess audio data
    audio, sample_rate = torchaudio.load("data/whisper/gettysburg.wav")
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        data = resampler(audio)
        data = audio.squeeze(0)

    # Create input tensor
    input_tensor = httpclient.InferInput("audio", data.shape, datatype="FP32")
    input_tensor.set_data_from_numpy(data.numpy())

    # Query the server
    response = client.infer(model_name="whisper_base", inputs=[input_tensor])

    # Get the output tensor
    output_tensor = response.as_numpy("transcription")

    print("Transcription:", output_tensor[0].decode("utf-8"))
