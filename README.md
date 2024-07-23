# traNsLatorLaB
The repository contains the source code of the python script, as well as notebooks for using parts extracted from the script via Colaboratory (.ipynb).

## Description

This Python script is designed to facilitate language translation, leveraging advanced machine learning models to accurately translate text from one language to another. It utilizes the `transformers` library to access pre-trained models for high-quality translations. The script supports translating text files or input from the standard input, making it versatile for various use cases.

## Features

- **Language Detection**: Automatically detects the language of the input text using both `langdetect` and a fastText model, ensuring accurate translation.
- **Translation**: Translates text to a specified target language using state-of-the-art models from Helsinki-NLP or Facebook's NLLB-200 distilled model, depending on availability and compatibility.
- **Device Compatibility**: Automatically detects and utilizes available computational resources, preferring GPU acceleration when available for faster processing.
- **Customizable Output**: Allows users to specify an output file for the translated text or prints the translation to the terminal if no file is specified.
- **Sentence Splitting**: Splits input text into manageable sentences or segments to ensure the translation's quality and coherence, especially for longer texts.

### Dependencies

- Python 3.x
- `transformers`
- `torch`
- `requests`
- `nltk`
- `pycountry`
- `langdetect`
- `sentencepiece`
- `fasttext`
- `sacremoses` (optional)

### Installing

1. Clone the repository to your local machine.

```bash
git clone https://github.com/r1cc4rd0m4zz4/traNsLatorLaB.git
```
 
2. Install the required Python packages using `pip`:

```bash
pip install transformers torch requests nltk pycountry langdetect fasttext sentencepiece sacremoses
```
or
```bash
pip install -r requirements.txt
```

3. Download and prepare any necessary models or data files as described in the script comments or documentation.

### Executing Program

1. Run the script from the command line, optionally specifying the input text file and target language, by default the script will translate to Italian, example:

```bash
pbpaste | python translatorlab.py [-o OUTPUT] [-l {it,en}] [txt_path] | pbcopy
```

2. For direct text input or to use the script in an interactive mode, follow the instructions provided in the script's comments or use the -h flag to access help:

```bash
python translatorlab.py -h
```

## Indemnity and disclaimer

Use of the TraNsLatorLaB machine translation template is at your own risk. The author of the code assumes no liability for any damage or loss resulting from the use of the template.

In addition, use of the template may be subject to local or international laws and regulations. It is the user's responsibility to verify that the use of the template complies with applicable laws and regulations.

Finally, the author of the code does not guarantee the security of the template or its compliance with privacy or data security regulations. It is your responsibility to ensure the security and privacy of your data and to use the template in compliance with applicable regulations.

## Acknowledgments

- Helsinki-NLP for the translation models
- Facebook AI for the NLLB-200 distilled model
- The transformers library by Hugging Face
- The PyTorch team for providing an open-source machine learning library for Python
- The requests library for providing a simple interface for making HTTP requests
- The Natural Language Toolkit (`nltk`) team for providing essential NLP tools and resources
- The pycountry library for providing ISO country code utilities and data
- The langdetect library for providing language detection capabilities
- The sentencepiece library for providing efficient subword tokenization
- The fasttext library for providing fast and accurate language identification
- The sacremoses library for providing tokenization and detokenization utilities
