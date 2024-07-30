import wandb
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


# runtime params
os.environ["WANDB_DISABLED"] = "false"

run_local = True
use_entire_dataset = True  # kept off to train on same size over all datasets
# select dataset
use_vp = False
use_cv = False
use_arxif = False
use_cornell_obj = True
use_cornell_subj = False

# check that only one dataset is selected
assert sum([use_vp, use_cv, use_arxif, use_cornell_obj, use_cornell_subj]) == 1

EARLYSTOPPING = False
NOMETRICS = False

# hyperparams
num_training_samples = 175000
num_validation_samples = 1500
output_model_name = "model_name"
num_train_epochs = 5
eval_save_steps = 25

# for tuning
# Best config: {'batch_size': 20, 'learning_rate': 0.00021067421106137532, 'lora_rank': 30}

batch_size = 20
gradient_accumulation_steps = batch_size // 8

learning_rate = 0.00021

lora_rank = 30

###############################################
# logging setup: to console and to file
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
timestap = time.time()
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(timestap))
handler = logging.FileHandler(f"kaggle/working/training_{timestamp}.log/")
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

if use_vp:
    output_path = "kaggle/working/" + output_model_name
    dataset_path = "kaggle/input/train-datasets/vp_mod_30.hf"
elif use_cv:
    output_path = "kaggle/working/" + output_model_name
    dataset_path = "kaggle/input/train-datasets/cv_mod_30.hf"
    
elif use_arxif:
    output_path = "kaggle/working" + output_model_name
    dataset_path = "kaggle/input/train-datasets/arxiv_abstracts"
elif use_cornell_obj:
    output_path = "kaggle/working/" + output_model_name
    dataset_path = "kaggle/input/train-datasets/cornell_obj"
elif use_cornell_subj:
    output_path = "kaggle/working/" + output_model_name
    dataset_path = "kaggle/input/train-datasets/cornell_sub"
working_dir = "kaggle/working/temp_data_sets/"

load_dotenv(override=True)  # refresh the environment variables, needed after changing the .env file!

# get tokens
hf_token = os.getenv("HF_TOKEN")
wandb_token = os.getenv("wandb_api_key")

# login into the clients
wandb.login(key=wandb_token)
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


# TODO validate this
# delete the temp_data_sets directory and all its contents
if run_local:
    shutil.rmtree("kaggle/working/temp_data_sets", ignore_errors=True)
    os.makedirs("kaggle/working/temp_data_sets", exist_ok=True)
else:
    shutil.rmtree("/kaggle/working/temp_data_sets", ignore_errors=True)
    os.makedirs("/kaggle/working/temp_data_sets", exist_ok=True)

# if run_local:
#     !rm -rf "kaggle/working/temp_data_sets"
#     !mkdir "kaggle/working/temp_data_sets"
# else:
#     !rm -rf "/kaggle/working/temp_data_sets"
#     !mkdir "/kaggle/working/temp_data_sets"

# if use_vp:  # not needed for cleaned datasets
#     text_field = "text"
# else:
#     text_field = "sentence"


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
# working_dir = "./"
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
elif use_cornell_obj:
    train_cache = working_dir + "cornell_obj_train.cache"
    val_cache = working_dir + "cornell_obj_valid.cache"
    test_cache = working_dir + "cornell_obj_test.cache"
elif use_cornell_subj:
    train_cache = working_dir + "cornell_subj_train.cache"
    val_cache = working_dir + "cornell_subj_valid.cache"
    test_cache = working_dir + "cornell_subj_test.cache"
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

num_devices = torch.cuda.device_count()

# define parameters
max_seq_length = max_length
gradient_accumulation_steps = gradient_accumulation_steps
warmup_steps = 2
learning_rate = learning_rate
fp16 = True
optim = "paged_adamw_8bit"
evaluation_strategy = "steps"
eval_steps = eval_save_steps
num_train_epochs = num_train_epochs
batch_size = batch_size
logging_steps = 1
save_strategy = "steps"
save_steps = eval_save_steps
hub_strategy = "every_save"
push_to_hub = True
hub_token = hf_token
hub_private_repo = True
metric_for_best_model = "eval_loss"
# greater_is_better=False # will default to False if you use eval_loss for metric for best model
load_best_model_at_end = True
lr_scheduler_type = "cosine"


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
    output_dir=output_path,
    report_to="wandb",
    save_strategy=save_strategy,
    save_steps=save_steps,
    hub_strategy=hub_strategy,
    push_to_hub=push_to_hub,
    hub_token=hub_token,
    hub_private_repo=True,
    metric_for_best_model=metric_for_best_model,
    load_best_model_at_end=load_best_model_at_end,
    lr_scheduler_type=lr_scheduler_type,

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
if use_cornell_obj:
    dataset = "Cornell_Objective"
if use_cornell_subj:
    dataset = "Cornell_Subjective"

config = {
    "max_seq_lentgh": max_seq_length,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "warmup_steps": warmup_steps,
    "learning_rate": learning_rate,
    "fp16": fp16,
    "optim": optim,
    "eval_strategy": evaluation_strategy,
    "per_device_train_batch_size": batch_size,
    "num_train_epochs": num_train_epochs,
    "logging_steps": logging_steps,
    "dataset": dataset,
    "Training samples used": num_training_samples,
    "Validation samples used": num_validation_samples,
    "project": "gemma2b_identifying_opinions_in_informative_text",
    "experiment_name": output_model_name,
    "learning_rate_scheduler": lr_scheduler_type,
}
wandb.init(config=config, project=config["project"], name=config["experiment_name"])

metric_functions = {
    # 'bleu': evaluate.load("bleu"),
    'rogue': evaluate.load("rouge"),  # only for final training
    # 'meteor': evaluate.load('meteor'), # TODO maybe add
    # "mauve": evaluate.load("mauve"),  # Needs atleast 78 points!!!
    # "perplexity": evaluate.load("perplexity", module_type="metric", hf_token=hf_token),
}


def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(axis=-1)


def compute_metrics(eval_preds):
    if NOMETRICS:
        return {}
    results = {}
    predictions, labels = eval_preds

    # -100 is typically used as a placeholder for padding or ignored tokens
    # and as we can not decode them we remove them
    preds = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute each metric
    for metric_name, metric_function in metric_functions.items():
        if metric_name == "perplexity":
            metric_result = metric_function.compute(
                model_id="google/gemma-2b",
                add_start_token=False,
                predictions=decoded_preds,
            )
        else:
            metric_result = metric_function.compute(
                predictions=decoded_preds, references=decoded_labels
            )
        # For BLEU score or other single scalar metrics, adapt the key - TODO do we want more from the metric?
        if metric_name == "bleu":
            results[f"{metric_name} score"] = metric_result[f"{metric_name}"]
        elif metric_name == "mauve":
            results[f"{metric_name} score"] = metric_result.mauve
        elif metric_name == "perplexity":
            results["mean perplexity score"] = metric_result["mean_perplexity"]
        elif metric_name == "rogue":
            results.update(metric_result)
    return results


class SaveIncumbentCallback(TrainerCallback):  # additional saving of best model to hub to keep track of best model separately
    def __init__(self, hub_model_id):
        self.hub_model_id = hub_model_id
        self.api = huggingface_hub.HfApi(token=hf_token)
        infos = self.api.whoami()
        self.username = infos["name"]
        self.best_metric = float('inf')
        self.repopath = f"{self.username}/{self.hub_model_id}"
        logging.debug(f"Token being used: {hf_token[:5]}...{hf_token[-5:]}")  # Print first and last 5 characters
        try:
            huggingface_hub.hf_api.create_repo(self.repopath, private=True, token=hf_token, repo_type="model", exist_ok=True)    
        except:
            logger.info(f"Model {self.repopath} already exists.")
            huggingface_hub.hf_api.create_repo(self.repopath, private=True, token=hf_token, repo_type="model", exist_ok=True)

    def on_evaluate(self, args, state, control, metrics, model, tokenizer, **kwargs):
        # Assuming you're tracking 'eval_loss', change this if you're using a different metric
        current_metric = metrics.get("eval_loss")
        
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            
            # Save the model locally
            output_dir = os.path.join(args.output_dir, "incumbent")
            model.save_pretrained(output_dir)
            
            # Push to Hub
            # refresh the api object
            self.api = huggingface_hub.HfApi(token=hf_token)
            self.api.upload_folder(
                folder_path=output_dir,
                repo_id=self.repopath,
                repo_type="model",
                commit_message=f"New best model with eval_loss: {current_metric}"
            )
            logger.info(f"Pushed new best model to {self.hub_model_id}")

incumbent_callback = SaveIncumbentCallback(hub_model_id=f"{output_model_name}_incumbent")

if EARLYSTOPPING:
    trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=sft_config,
    peft_config=lora_config,
    tokenizer=tokenizer,
    callbacks=[incumbent_callback, early_stopping_callback],
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
else:
    trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=sft_config,
    peft_config=lora_config,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[incumbent_callback],
    )
trainer.train()
trainer.push_to_hub()
