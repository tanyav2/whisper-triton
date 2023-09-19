# Test script to run the pipeline once it has been loaded in triton
import tritonclient.http as httpclient
import numpy as np
import argparse


def main(model_name, file_path):
    # Setting up client
    client = httpclient.InferenceServerClient(url="localhost:8000")

    buf = open(file_path, "rb").read()
    arr = np.array([buf], dtype=np.object_)

    # Create input tensor
    input_tensor = httpclient.InferInput("audio", arr.shape, datatype="BYTES")
    input_tensor.set_data_from_numpy(arr)

    # Query the server
    response = client.infer(model_name=model_name, inputs=[input_tensor])

    # Get the output tensor
    output_tensor = response.as_numpy("transcription")

    print("Transcription:", output_tensor[0].decode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        default="whisper_base",
        help="Name of the model to use. Default is 'whisper_base'."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the audio file to be transcribed. Acceptable formats are m4a, mp3, mp4, mpeg, mpga, wav, and webm."
    )
    args = parser.parse_args()
    main(args.model_name, args.file)
