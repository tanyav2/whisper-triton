#!/bin/bash

pip install optimum[exporters]
optimum-cli export onnx --model="openai/whisper-base" --task automatic-speech-recognition whisper-base-onnx/