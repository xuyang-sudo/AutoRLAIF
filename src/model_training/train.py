import os
import torch
from transformers import (
    BitsAndBytesConfig,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
    TensorBoardCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset, concatenate_datasets

from src.configs.config import Config
from src.data_processing.custom_tokenizer import CustomTokenizer
from src.model_evaluation.metrics import compute_metrics
from src.utils.callbacks import CustomCallback
# from src.model_training.ema import LoRAEMACallback
# from src.model_training.rdrop import RDropTrainer

def main():
    config = Config()

    # 加载模型和分词器
    tokenizer = GemmaTokenizerFast.from_pretrained(config.checkpoint)
    model = Gemma2ForSequenceClassification.from_pretrained(
        config.checkpoint,
        num_labels=3,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # LoRA 配置
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        layers_to_transform=[i for i in range(42) if i >= config.freeze_layers],
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)

    # 加载数据集
    # 使用 load_dataset 和 concatenate_datasets 加载并合并数据集
    from datasets import load_dataset, concatenate_datasets

    # 加载各个数据集
    dataset1 = load_dataset('csv', data_files={'train': './data/lmsys-arena-human-preference-55k/*.csv'})
    dataset2 = load_dataset('csv', data_files={'train': './data/lmsys-chatbot_arena_conversations-33k/*.csv'})
    # 如果需要使用第三个数据集，可以取消注释以下代码
    # dataset3 = load_dataset('csv', data_files={'train': './data/lmsys-Pairs-generated-from-lmsys-1M-dataset/*.csv'})

    # 合并数据集
    datasets_to_concat = [dataset1['train'], dataset2['train']]
    # 如果使用第三个数据集，取消以下注释
    # datasets_to_concat.append(dataset3['train'])

    raw_dataset = concatenate_datasets(datasets_to_concat)

    # 应用自定义的 tokenizer
    tokenizer_fn = CustomTokenizer(tokenizer, max_length=config.max_length)
    ds = raw_dataset.map(tokenizer_fn, batched=True)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        report_to="none",
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=500,
        optim=config.optim_type,
        bf16=True,
        learning_rate=config.lr,
        warmup_steps=config.warmup_steps,
        gradient_checkpointing=True,
        metric_for_best_model="log_loss",
        deepspeed=None,
        save_only_model=True,
        lr_scheduler_type='linear',
    )

    # 创建 Trainer
    trainer = Trainer(
        args=training_args, 
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=[TensorBoardCallback(), CustomCallback()],
        # compute_metrics=compute_metrics,
    )

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
