# Efficient Story Generation with Small Language Models : Training-GPT-2-on-the-TinyStories-Dataset


The project focuses on efficient narrative generation using relatively small models and evaluates their ability to generate coherent stories.
The pipeline includes:
Dataset preparation
BPE tokenizer training
GPT-2 model training
Story generation
Evaluation using metrics
The system is designed to be fully reproducible and configurable via YAML configuration files.

```Plaintext
tinystories-gpt2/
│
├── configs/                       # YAML configuration files
│   ├── tokenizers/
│   │   └── tinystories_1M.yaml    # tokenizer + dataset configuration
│   │    ...
│   │   
│   └── training/
│     └── train_tinystories_1M.yaml # training hyperparameters
│      ...
│
├── data/                          # evaluation datasets
│   ├── prompts.txt                # prompts used for generation
│   └── references.txt             # reference stories for evaluation
│
├── results/                       # saved experiment outputs
│   └── gpt2_medium/
│       ├── model/                 # trained GPT-2 model checkpoints: medium
│       └── tokenizer/             # trained BPE tokenizer
│   └── gpt2_small/
│       ├── model/                 # trained GPT-2 model checkpoints : smmall
│       └── tokenizer/ 
├── bpe_tokenizer_utils.py         # BPE tokenizer training and loading utilities
├── config_utils.py                # helper functions for loading YAML configs
│
├── gpt2.py                        # dataset preparation and tokenizer training
├── gpt2_trainer.py                # GPT-2 training pipeline using HuggingFace Trainer
├── gpt2main.py                    # main entry point for tokenizer creation and training
│
├── generate.py                    # text generation script using trained model
├── evaluation.py                  # evaluation metrics 
├── prepare_eval_data.py           # script to create evaluation prompts and references
│
├── data.txt                       # sample TinyStories text examples
├── eval_test.json ...             # example evaluation results
│
└── README.md                      # project documentation

```

## Project Workflow
### 1. Prepare Evaluation Data
This script extracts prompts and reference stories from TinyStories.
File: prepare_eval_data.py It loads the TinyStories dataset and writes:
data/prompts.txt
data/references.txt

### 2. Build Tokenizer and Training Dataset
This stage in file: gpt2.py:
• downloads TinyStories
• trains a BPE tokenizer
• tokenizes the dataset
• creates a language modeling dataset
 

 The class TextGenatate loads TinyStories from HuggingFace, trains or loads a BPE tokenizer, tokenizes the dataset andgroups tokens into fixed blocks for LM training. 

Tokenizer training is implemented in: bpe_tokenizer_utils.py

### 4. Train GPT-2 Model
Training is implemented in file: gpt2_trainer.py


It loads the tokenized dataset, builds a GPT-2 model from scratch, loads configuration from YAML and runs HuggingFace Trainer.


### 4. Generate Stories
After training, we can generate text, loads the trained GPT-2 model, loads the tokenizer, samples text using top-k / top-p sampling.


### 5. Evaluate Model Performance
Evaluation is implemented in evaluation.py. The evaluation compares generated stories with reference stories using metrics.
Run evaluation:
```bash
python evaluation.py \
    --model_dir results/gpt2_medium/model \
    --tokenizer_dir results/gpt2_medium/tokenizer \
    --prompts data/prompts.txt \
    --references data/references.txt
```


## Future Work
Possible extensions:
training larger GPT-2 variants
comparison with Mamba architectures
scaling experiments with different dataset sizes
improved evaluation metrics

## References
**TinyStories Dataset**: Ronen Eldan and Yuanzhi Li
TinyStories: How Small Can Language Models Be? 2023

**GPT-2**: Radford et al. Language Models are Unsupervised Multitask Learners OpenAI , 2019    
