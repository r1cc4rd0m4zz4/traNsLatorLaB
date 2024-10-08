#!/usr/bin/env python3
import argparse
import nltk
from nltk.tokenize import sent_tokenize
from nltk.data import find
import fasttext
from langdetect import DetectorFactory, detect_langs
import sys
import os
import requests
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import pycountry
import textwrap
import shutil
import time
import tiktoken
import string

def get_ft_model():
    try:
        # Define the path where you want to save the model
        model_path = Path.home() / ".fasttext_models/lid.176.ftz" # Faster but less accurate model
        #model_path = Path.home() / ".fasttext_models/lid218e.bin" # Slower but more accurate model
        # Check if the model already exists, otherwise download it
        if not model_path.exists():
            print("fastText model not found. Starting download...")
            os.makedirs(model_path.parent, exist_ok=True)  # Create the directory if it does not exist
            download_file("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz", model_path) # Fast but less accurate
            #download_file("https://tinyurl.com/nllblid218e", model_path) # Slower but more accurate
        # Load the fastText model
        ft_model = fasttext.load_model(str(model_path))
    except Exception as e:
        print(f"Error loading the fastText model: {e}")
        ft_model = None
        sys.exit(1)
    return ft_model

def get_nltk_punkt():
    try:
        # Check if the 'punkt' package has already been downloaded
        find('tokenizers/punkt')
    except LookupError:
        try:
            # Attempt to download the 'punkt' package
            nltk.download('punkt')
        except Exception as e:
            # Handle the download error
            print(f"Error downloading the 'punkt' package: {e}")
            sys.exit(1)

def download_file(url, save_path):
    """
    Download the file from `url` and save it as `save_path`.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raises an exception for failed responses
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"Model downloaded and saved in {save_path}")

def detect_language_fasttext(text:str, ft_model=None):
    try:
        if ft_model:
            predictions = ft_model.predict(text)
            #print(predictions)
            return predictions[0][0].split("__label__")[1]
        else:
            return None
    except Exception as e:
        print(f"Error in fasttext language detection: {e}")
        return None

def detect_language_langdetect(text):
    try:
        DetectorFactory.seed = 0
        predictions = detect_langs(text)
        #print(predictions)
        return predictions[0].lang
    except Exception as e:
        print(f"Error in langdetect language detection: {e}")
        return None
    
def split_sentences(text, max_length=512):
    get_nltk_punkt()
    sentences = sent_tokenize(text)
    enc = tiktoken.get_encoding("gpt2")
    short_sentences = []
    for sentence in sentences:
        tokens = enc.encode(sentence)
        while len(tokens) > max_length:
            current_sentence = tokens[:max_length]
            # Ensure we don't split in the middle of a word or punctuation
            if len(current_sentence) < len(tokens):
                split_point = max_length
                while split_point > 0 and not enc.decode([tokens[split_point]]).isspace():
                    split_point -= 1
                if split_point == 0:
                    split_point = max_length
                current_sentence = tokens[:split_point]
            short_sentences.append(enc.decode(current_sentence).strip())
            tokens = tokens[len(current_sentence):]
        short_sentences.append(enc.decode(tokens).strip())
    return short_sentences

def detect_language(text, ft_model=None):
    lang = detect_language_langdetect(text) or detect_language_fasttext(text, ft_model) or "Unknown" 
    return lang

def save_text(text, base_file_path):
    """
    Saves the given text to a file, creating a new file if one already exists.
    """
    index = 1
    file_path = base_file_path
    while os.path.exists(file_path + ".txt"):
        file_path = f"{base_file_path}_{index}"
        index += 1
    file_path += ".txt"
    with open(file_path, "w") as file:
        file.write(text)
    print(f"Text saved in: {file_path}")

def load_text(base_file_path):
    """
    Loads and returns the text from a file if it exists.
    """
    if os.path.exists(base_file_path):
        with open(base_file_path, 'r') as file:
            text = file.read()
            return text
    else:
        sys.exit(f"The file {base_file_path} does not exist.")

def get_available_device():
    """
    Returns the available device for computation ('cuda:0' for GPU, 'cpu' for CPU).
    """
    if torch.cuda.is_available():
        return "cuda:0"
    #elif torch.backends.mps.is_available():
    #    # If MPS is available (for Apple devices with M1/M2 chip), use MPS
    #    return "mps"
    else:
        return "cpu"

def convert_language_code(lang_iso639_1):
    """
    Converts a two-letter ISO 639-1 language code to a custom format using the three-letter ISO 639-3 code.
    """
    lang_iso639_3 = pycountry.languages.get(alpha_2=lang_iso639_1).alpha_3
    custom_format = f"{lang_iso639_3}_Latn"
    return custom_format

def translate_text(text, target_lang, device="cpu", ft_model=None, ts_model=None):
    """
    Translates the given text to the target language using a specified device and fastText model for language detection.
    """
    # Check if the text contains only punctuation symbols
    if all(char in string.punctuation for char in text):
        return text
    origin_lang = detect_language(text, ft_model)
    input_lang = convert_language_code(origin_lang)
    output_lang = convert_language_code(target_lang)
    
    if (ts_model is  None) or (ts_model == "opus"):        
        model_names = [
            f'Helsinki-NLP/opus-mt-{origin_lang}-{target_lang}',  # First attempt with a specific model
            'facebook/nllb-200-distilled-600M'  # Fallback model
        ]
    elif (ts_model == "nllb") or (ts_model == "nllb-d600"):
        model_names = [
            'facebook/nllb-200-distilled-600M',  # First attempt with a specific model
            f'Helsinki-NLP/opus-mt-{origin_lang}-{target_lang}'  # Fallback model
        ]    
    elif (ts_model == "nllb-d1.3"):
        model_names = [
            'facebook/nllb-200-distilled-1.3B',  # First attempt with a specific model
            f'Helsinki-NLP/opus-mt-{origin_lang}-{target_lang}'  # Fallback model
        ]
    elif (ts_model == "nllb-1.3"):
        model_names = [
            'facebook/nllb-200-1.3B',  # First attempt with a specific model
            f'Helsinki-NLP/opus-mt-{origin_lang}-{target_lang}'  # Fallback model
        ]
    elif (ts_model == "nllb-3.3"):
        model_names = [
            'facebook/nllb-200-3.3B',  # First attempt with a specific model
            f'Helsinki-NLP/opus-mt-{origin_lang}-{target_lang}'  # Fallback model
        ]
    elif (ts_model == "m2m") or (ts_model == "m2m-418"):
        model_names = [
            'facebook/m2m100_418M',  # First attempt with a specific model
            f'Helsinki-NLP/opus-mt-{origin_lang}-{target_lang}'  # Fallback model
        ]
    elif (ts_model == "m2m-1.2"):
        model_names = [
            'facebook/m2m100_1.2B',  # First attempt with a specific model
            f'Helsinki-NLP/opus-mt-{origin_lang}-{target_lang}'  # Fallback model
        ]
    else:
        raise ValueError("Invalid model value. Expected 'opus' or 'nllb' or 'm2m'.")
    for model_name in model_names:
        try:
            #print(model_name + " model")
            if output_lang != input_lang:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False, clean_up_tokenization_spaces=True)
                translation_pipeline = pipeline(
                    'translation',
                    model=model,
                    tokenizer=tokenizer,
                    src_lang=input_lang,
                    tgt_lang=output_lang,
                    max_length=400,
                    device=device
                )
                output = translation_pipeline(text)
                if output != text:                    
                    return output[0]['translation_text']
                else:
                    #print(f"Translating from {origin_lang} to {target_lang}, text: {text}")
                    continue
            else:
                return text
        except Exception as e:
            #print(f"Error during the loading of {model_name}: {e}")
            continue
    raise RuntimeError("Unable to load both the specific and fallback models")

def print_in_blocks(text):
    """
    Prints the text in blocks that fit the terminal width.
    """
    terminal_size = shutil.get_terminal_size()
    terminal_width = terminal_size.columns
    wrapped_text = textwrap.fill(text, terminal_width)
    print(wrapped_text)
        
def main():
    """
    Main function to translate text from a file or standard input to another language.
    """
    parser = argparse.ArgumentParser(description="Translates a text to another language.")
    parser.add_argument("txt_path", nargs='?', default=None, help="The path of the txt file from which to translate text.")
    parser.add_argument("-o", "--output", help="The path of the output text file. If not specified, the text will be printed to the terminal.")
    parser.add_argument("-l", "--lang", choices=["it", "en"], default="it", help="The text translate in: 'Italian' (default) or 'English'.")
    parser.add_argument("-m", "--model", choices=["opus", "m2m", "m2m-418", "m2m-1.2", "nllb", "nllb-d600", "nllb-1.3", "nllb-d1.3", "nllb-3.3"], default="opus", help="The translator model: 'Helsinki-NLP' (default) or 'Facebook/m2m100' or 'Facebook/nllb'.")
    parser.add_argument("-s", "--stream", action="store_true", help="Stream the translated text instead of printing it all at once.")
    
    args = parser.parse_args()
    
    if not args.txt_path:
        #text = "Come ti chiamo? Je m'appelle Marie. ¿Cómo te llamas? Ich heiße Peter. ¿Y tú? Mi chiamo Giovanni. Et toi? He didn't want to come with me." # Only for test text
        text = sys.stdin.read()
    else:
        text = load_text(args.txt_path)              
    segments = split_sentences(text, 400)
    device = get_available_device()
    ft_model = get_ft_model()
    translated_segments = []
    for segment in segments:
        #print(segment)
        translated_segment = translate_text(segment, args.lang, device, ft_model, args.model)
        #print(translated_segment)
        if args.stream:
            words = translated_segment.split()
            for word in words:
                print(word, end=' ', flush=True)
                time.sleep(0.02) # Delay to simulate streaming
        translated_segments.append(translated_segment)
    text = ' '.join(translated_segments)
    if args.output:
        save_text(text, args.output)
    else:
        if not args.stream:
            #print_in_blocks(text)
            print(text)

if __name__ == "__main__":
    main()
