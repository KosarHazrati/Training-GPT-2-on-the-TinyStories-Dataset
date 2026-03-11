# Efficient Story Generation with Small Language Models : Training-GPT-2-on-the-TinyStories-Dataset


The project focuses on efficient narrative generation using relatively small models and evaluates their ability to generate coherent stories.
The pipeline includes:
1. **Tokenizer training**
   - A custom BPE tokenizer is trained on a subset of TinyStories.
2. **Dataset preparation**
   - The raw text corpus is tokenized and grouped into fixed-length blocks for causal language modeling.
3. **Model training**
   - A GPT-2 style language model is initialized from scratch and trained with Hugging Face `Trainer`.
4. **Story generation**
   - The trained model can generate continuations from a given prompt.
5. **Evaluation**
   - Generated stories are compared against references using metrics.
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
│       ├── model/                 # trained GPT-2 model checkpoints : small
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


## Result
## Result: 
We observe

-  View Full Experiment [Table](https://github.com/KosarHazrati/Training-GPT-2-on-the-TinyStories-Dataset/blob/main/wandb_export_2026-03-09T11_31_59.190%2B01_00.csv) 


## Future Work
Possible extensions are training larger GPT-2 variants, comparison with Mamba architectures, scaling experiments with different dataset sizes and improved evaluation metrics.

## References
 **TinyStories:** Eldan, R., & Li, Y. (2023). [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759).

**GPT-2**: Radford et al. (2019 ). [Language Models are Unsupervised Multitask Learners OpenAI ]  (https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
