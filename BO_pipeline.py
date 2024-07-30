import logging
import time
import numpy as np
import neps
import huggingface_hub
import os
from dotenv import load_dotenv
from peft import LoraConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from datasets import load_dataset, load_from_disk
from trl import SFTConfig
from transformers.trainer_callback import EarlyStoppingCallback
import evaluate
import numpy as np
from trl import SFTTrainer
import shutil
import logging
import sys
import time
from pathlib import Path
os.environ["WANDB_DISABLED"] = "true"  # don't log to wandb

# runtime params
use_entire_dataset = False  # kept off to train on same size over all datasets
use_vp = True
use_cv = False
use_arxif = False

EARLYSTOPPING = True
INCUMBENT = False

RUNHPO = True  # set to False to run with fixed hyperparameters specified at end of file

# hyperparams
num_training_samples = 20000  # real value
# num_training_samples = 200  # test value to check if pipeline works
num_validation_samples = 1500  # real value
# num_validation_samples = 15  # test value to check if pipeline works
output_model_name = "Model_Name"
num_train_epochs = 1
eval_steps = 50

###############################################
# logging setup: to console and to file
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
timestap = time.time()
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(timestap))
handler = logging.FileHandler(f"results/{output_model_name}/training_{timestamp}.log/")  # log in NePS folder
handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console_handler)

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

###############################################

def run_pipeline(pipeline_directory: Path, 
                 batch_size, learning_rate, lora_rank,
                 num_training_samples=num_training_samples,
                 num_validation_samples=num_validation_samples,
                    use_entire_dataset=use_entire_dataset,
                    use_vp=use_vp,
                    use_cv=use_cv,
                    use_arxif=use_arxif,
                    output_model_name=output_model_name,
                    num_train_epochs=num_train_epochs,
                    eval_steps=eval_steps,
                 ):
    """
    Wrapper function to run the pipeline with the given hyperparameters.
    Args:
    Searched parameters:
        batch_size: Batch size
        learning_rate: Learning rate
        lora_rank: Rank of the LoRA matrix
    Manual, fixed parameters:
        pipeline_directory: Path to the directory where the pipeline should be run
        num_training_samples: Number of training samples
        num_validation_samples: Number of validation samples
        use_entire_dataset: Whether to use the entire dataset
        use_vp: Whether to use the VoxPopuli dataset
        use_cv: Whether to use the CommonVoice dataset
        use_arxif: Whether to use the Arxiv dataset
        output_model_name: Name of the output model
        num_train_epochs: Number of training epochs
        eval_steps: Evaluation steps

    Returns:
        loss: The loss of the model on the test set

    """
    
    gradient_accumulation_steps = batch_size // 8
    
    
    if use_vp:
        output_path = "kaggle/working/" + output_model_name
        dataset_path = "kaggle/input/train-datasets/vp_mod_30.hf"
    elif use_cv:
        output_path = "kaggle/working/" + output_model_name
        dataset_path = "kaggle/input/train-datasets/cv_mod_30.hf"
    elif use_arxif:
        output_path = "kaggle/working" + output_model_name
        dataset_path = "kaggle/input/train-datasets/arxiv_abstracts"
    working_dir = "kaggle/working/temp_data_sets/"

    load_dotenv(override=True)  # refresh the environment variables, needed after changing the .env file!

    # get tokens
    hf_token = os.getenv("HF_TOKEN")
    huggingface_hub.login(token=hf_token)

    # load the dataset
    data = load_from_disk(dataset_path)
    if use_entire_dataset:
        data_train = data["train"]
        data_validation = data["validation"]
        data_test = data["test"]
        num_training_samples = "all"
        num_validation_samples = "all"
    else:
        data_train = data["train"].select(range(num_training_samples))
        data_validation = data["validation"].select(range(num_validation_samples))
        data_test = data["test"]

    lora_config = LoraConfig(
        r=lora_rank,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model_id = "google/gemma-2b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=hf_token, padding_side="right"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map={"": 0}, token=hf_token
    )

    shutil.rmtree("kaggle/working/temp_data_sets", ignore_errors=True)
    os.makedirs("kaggle/working/temp_data_sets", exist_ok=True)


    def tokenize_and_cache(data, tokenizer, max_length, cache_file_name):
        return data.map(
            lambda examples: tokenizer(
                examples["text"], # type: ignore
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
                return_tensors="pt",
            ),
            batched=True,
            cache_file_name=cache_file_name,
        )
    max_length = 16

    # Paths to cache files
    working_dir = "./"
    if use_vp:
        train_cache = working_dir + "vp_train.cache"
        val_cache = working_dir + "vp_valid.cache"
        test_cache = working_dir + "vp_test.cache"
    elif use_cv:
        train_cache = working_dir + "cv13_train.cache"
        val_cache = working_dir + "cv13_valid.cache"
        test_cache = working_dir + "cv13_test.cache"
    elif use_arxif:
        train_cache = working_dir + "arxiv_train.cache"
        val_cache = working_dir + "arxiv_valid.cache"
        test_cache = working_dir + "arxiv_test.cache"
    # clear cache
    try:
        os.remove(train_cache)
        os.remove(val_cache)
        os.remove(test_cache)
    
    except:
        shutil.rmtree(train_cache, ignore_errors=True)
        shutil.rmtree(val_cache, ignore_errors=True)
        shutil.rmtree(test_cache, ignore_errors=True)


    # Tokenize and cache
    train_data = tokenize_and_cache(data_train, tokenizer, max_length, train_cache)
    val_data = tokenize_and_cache(data_validation, tokenizer, max_length, val_cache)
    test_data = tokenize_and_cache(data_test, tokenizer, max_length, test_cache)

    
    # define parameters
    max_seq_length = max_length
    gradient_accumulation_steps = gradient_accumulation_steps
    warmup_steps = 2
    learning_rate = learning_rate
    fp16 = True
    optim = "paged_adamw_8bit"
    evaluation_strategy = "steps"
    eval_steps = eval_steps
    num_train_epochs = num_train_epochs
    batch_size = batch_size
    logging_steps = 1
    metric_for_best_model = "eval_loss"
    # greater_is_better=False # will default to False if you use eval_loss for metric for best model
    load_best_model_at_end = True
    lr_scheduler_type = "cosine"

    # generate unique output path to fix parallelization issue    

    sft_config = SFTConfig(
        max_seq_length=max_seq_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        optim=optim,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps,  # during real training, set to reasonable number. for debugging, prints every time
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        output_dir=pipeline_directory,
        metric_for_best_model=metric_for_best_model,
        load_best_model_at_end=load_best_model_at_end,
        lr_scheduler_type=lr_scheduler_type,
        report_to="none",

    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,  # Number of evaluations with no improvement after which training will be stopped
        early_stopping_threshold=0.01,  # Minimum change to qualify as an improvement
    )


    if use_vp:
        dataset = "VoxPopuli"
    if use_cv:
        dataset = "CommonVoice"
    if use_arxif:
        dataset = "Arxiv"


    if EARLYSTOPPING:
        trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=sft_config,
        peft_config=lora_config,
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback],
        # compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # callbacks=[SamplingCallback(tokenizer)],
        )
    else:
        trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=sft_config,
        peft_config=lora_config,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
    trainer.train()

    # evaluate
    metrics = trainer.evaluate(eval_dataset=test_data)
    loss = metrics["eval_loss"]
    logger.info(f"Loss: {loss}")
    
    return loss

###############################################
# NePS 
if RUNHPO:
    pipeline_space = dict(  # the searchspace for the pipeline
        batch_size=neps.IntegerParameter(lower=8, upper=32),  # gradient_accumulation_steps: %8
        learning_rate=neps.FloatParameter(lower=1e-5, upper=5e-3, log=True),
        lora_rank=neps.IntegerParameter(lower=1, upper=32),
    )

    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory=f"results/{output_model_name}",
        searcher="bayesian_optimization",
        random_interleave_prob= 0.1,
        loss_value_on_error=20,
        cost_value_on_error=20,
        post_run_summary=True,
        max_evaluations_total=30,
    )
else:
    params={
        "batch_size": 18,
        "learning_rate": 0.0035500067298605306,
        "lora_rank": 32
    }
    run_pipeline(**params)
