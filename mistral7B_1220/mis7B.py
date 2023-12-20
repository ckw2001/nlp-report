import os
# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='0'

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, get_peft_model
from transformers import (
    MistralForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser, 
    TrainingArguments, 
    BitsAndBytesConfig, 
    Trainer
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from accelerate import Accelerator


# Define and parse arguments.
@dataclass
class ScriptArguments:

    num_labels: Optional[int] = field(default=54, metadata={"help": "number of labels"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="/data/huggingface/transformers/mistralai/Mistral-7B-v0.1",
        metadata={"help": "the location of the pretrained model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=30, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})  # TODO

    per_device_train_batch_size: Optional[int] = field(default=8, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    load_in_4bit: Optional[bool] = field(
        default=True, metadata={"help": "load in 4bit"}
    )
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "load in 8bit"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    num_epochs: Optional[int] = field(default=3, metadata={"help": "max number of training epochs"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=10, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=10, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 2000 samples"})
    report_to: Optional[str] = field(
        default="tensorboard",
        metadata={
            "help": 'The list of integrations to report the results and logs to'
        },
    )


@dataclass
class MyCollator:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None

    def __call__(self, features):
        texts = [feature['text'] for feature in features]
        toked = self.tokenizer(texts, padding=self.padding, truncation=True, max_length=self.max_length, return_tensors='pt')
        labels = []
        for feature in features:
            labels.append([float(_) for _ in list(feature['targets'])])
        labels = torch.tensor(labels)
        return {
            'input_ids': toked.input_ids,
            'attention_mask': toked.attention_mask,
            'labels': labels
        }
    
class MyTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = True
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)
        
        logits = logits[0]
        labels = inputs['labels']
        return (loss, logits, labels)



if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    if script_args.sanity_check:
        script_args.num_epochs = 1
        script_args.warmup_steps = 2
        script_args.logging_steps = 1
        script_args.save_steps = 1
        script_args.eval_steps = 1
        script_args.gradient_accumulation_steps = 8

    # 1. load a pretrained model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    if script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
    elif script_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model = MistralForSequenceClassification.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
        problem_type="multi_label_classification",
        num_labels=script_args.num_labels,
        pad_token_id=tokenizer.eos_token_id
    )
    model.config.use_cache = False

    # 2. Load dataset
    raw_datasets = load_dataset(
        'json', 
        data_files={
            'train': "./data/train.json",
            'dev': "./data/dev.json",
            'test': "./data/test.json"
        },
    )

    def preprocess_function(examples):
        if isinstance(examples['text'], str):
            examples['text'] = examples['text'] + " " + tokenizer.eos_token
            return examples
        
        new_text = []
        for text in examples['text']:
            new_text.append(text + " " + tokenizer.eos_token)
        examples['text'] = new_text
        return examples
    
    for split in raw_datasets.keys():  
        if script_args.sanity_check:
            if split != 'train':
                raw_datasets[split] = raw_datasets[split].select(range(10))
            else:
                raw_datasets[split] = raw_datasets[split].select(range(4000))
        raw_datasets[split] = raw_datasets[split].map(
            preprocess_function,
            batched=True,
            num_proc=4
        )
    print(raw_datasets)

    
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()


    args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        evaluation_strategy='steps',
        eval_steps=script_args.eval_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant":False},
        learning_rate=script_args.learning_rate,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        weight_decay=script_args.weight_decay,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
    )

    data_collator = MyCollator(
        tokenizer,
        max_length=script_args.max_length
    )


    def compute_metrics(eval_pred):
        rank = Accelerator().local_process_index
        predictions, labels = eval_pred
        predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
        predictions = np.where(predictions > 0.5, 1, 0)
        
        precision = precision_score(labels, predictions, average='micro')
        recall = recall_score(labels, predictions, average='micro')
        f1 = f1_score(labels, predictions, average='micro')

        if rank == 0:
            step = trainer.state.global_step
            print(f'step: {trainer.state.global_step}, f1_score = {f1}')
            np.save(f'{script_args.output_dir}/predres/step_{step}.npy', np.stack([predictions, labels]))

        return {'precision': precision, 'recall': recall, 'f1': f1}


    trainer = MyTrainer(
        model=model,
        args=args,
        train_dataset=raw_datasets['train'],
        eval_dataset=raw_datasets['dev'],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
