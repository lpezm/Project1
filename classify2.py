#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import Interpreter

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--length', help='File with the number of characters', type=int)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    if args.length is None:
        print("Please specify the length")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    #with tflite.device('/cpu:0'):
    with open(args.output, 'w') as output_file:
        interpreter = tflite.Interpreter(model_path=args.model_name)

        for x in os.listdir(args.captcha_dir):
            # load image and preprocess it
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            image = numpy.array(rgb_data) / 255.0
            (c, h, w) = image.shape
            image = image.reshape([-1, c, h, w])
            image = numpy.float32(image)
            interpreter.set_tensor(input_details[0]['index'], image)
            out = ""
            interpreter.invoke()
            #procesa la imagen de una pero hay que sacarle caracter a caracter
            for i in range(0,5):
                output_data = numpy.squeeze(interpreter.get_tensor(output_details[i]['index']))
                output = numpy.argmax(output_data)
                out = out + captcha_symbols[output]

            output_file.write(x + "," + out)
            output_file.write("\n")
if __name__ == '__main__':
    main()
