# AutoRLAIF: Automated Reinforcement Learning from AI Feedback for Large Language Models

**AutoRLAIF** is a cutting-edge framework designed to revolutionize the fine-tuning of large language models through **Reinforcement Learning from AI Feedback (RLAIF)**. By automating the supervised fine-tuning (SFT) process, AutoRLAIF eliminates the need for extensive manual intervention, enhancing both efficiency and performance in developing sophisticated AI-driven conversational systems.

## 🚀 **Key Features**

- **Automated Fine-Tuning**: Leverages RLAIF to autonomously refine language models based on AI-generated feedback, minimizing the reliance on human supervision.
- **High-Efficiency Training**: Utilizes advanced techniques such as **QLoRA** and **Parameter-Efficient Fine-Tuning (PEFT)** to optimize training speed and resource utilization.
- **Data Integration**: Combines multiple high-quality datasets, including:
  - **lmsys-arena-human-preference-55k**: Comprehensive human preference data.
  - **lmsys-chatbot_arena_conversations-33k**: Extensive chatbot conversation logs.
  - **lmsys-Pairs-generated-from-lmsys-1M-dataset**: Large-scale AI-generated data pairs.
- **Advanced Training Techniques**:
  - **LoRA (Low-Rank Adaptation)**: Enhances model adaptability with minimal parameter updates.
  - **EMA (Exponential Moving Average)**: Stabilizes training by maintaining a moving average of model parameters.
  - **R-Drop**: Improves model robustness through regularization techniques.
- **Flexible Inference**: Implements **Test-Time Augmentation (TTA)** to ensure consistent and accurate predictions during the inference phase.
- **Scalable Architecture**: Designed to handle large-scale datasets and models, making it suitable for extensive AI applications across various domains.

## 🛠️ **Technologies Used**

- **Deep Learning Frameworks**: [PyTorch](https://pytorch.org/), Transformers
- **Fine-Tuning Tools**: [QLoRA](https://github.com/artidoro/qlora), [PEFT](https://github.com/huggingface/peft)
- **Data Processing**: Datasets, [NumPy](https://numpy.org/), Pandas
- **Optimization Tools**: [BitsAndBytes](https://github.com/facebookresearch/bitsandbytes), [DeepSpeed](https://www.deepspeed.ai/), [Scikit-learn](https://scikit-learn.org/)
- **Others**: [Matplotlib](https://matplotlib.org/), Seaborn

## 🎯 **Applications**

AutoRLAIF is ideal for developers, researchers, and organizations aiming to enhance their language models with minimal manual effort. Key applications include:

- **AI-Driven Chatbots**: Develop intelligent conversational agents that understand and respond to user preferences accurately.
- **User Preference Prediction**: Implement systems that can predict and adapt to user preferences in real-time.
- **Content Generation**: Create high-quality, contextually relevant content across various platforms and industries.
- **Research and Development**: Facilitate advanced research in natural language processing and machine learning by providing a robust framework for model fine-tuning.

## 📂 **Directory Structure**

```
css
AutoRLAIF/
├── README.md
├── LICENSE
├── data/
│   ├── lmsys-arena-human-preference-55k/
│   │   └── train.csv
│   ├── lmsys-chatbot_arena_conversations-33k/
│   │   └── train.csv
│   └── lmsys-Pairs-generated-from-lmsys-1M-dataset/
│       └── train.csv
├── src/
│   ├── data_processing/
│   │   └── custom_tokenizer.py
│   ├── model_training/
│   │   ├── train.py
│   │   ├── ema.py
│   │   └── rdrop.py
│   ├── model_evaluation/
│   │   └── metrics.py
│   ├── inference/
│   │   └── inference.py
│   ├── configs/
│   │   └── config.py
│   └── utils/
│       └── callbacks.py
├── doc/
│   └── Kaggle_Large_Model_Competition_Technical_Report.md
├── examples/
│   └── example_usage.ipynb
└── requirements.txt
```

## 📦 **Installation**

1. **Clone the Repository**

   ```
   bash
   git clone https://github.com/your_username/AutoRLAIF.git
   cd AutoRLAIF
   ```

2. **Install Dependencies**

   ```
   bash
   pip install -r requirements.txt
   ```

3. **Download Models and Datasets**

   - **Pre-trained Model**: Download the pre-trained **Gemma-2-9b-it** model and place it in the `./pretrained_models/gemma-2-9b-it-4bit` directory.
   - **Datasets**: Download and extract the required datasets into the `data/` directory, ensuring the directory structure matches the one outlined above.

## 🏃 **Usage**

### 🔥 **Training the Model**

Navigate to the `src/model_training/` directory and execute the training script:

```
bash
python train.py
```

**Configuration**: Modify training parameters in `src/configs/config.py` as needed.

### 🧠 **Model Inference**

After training, navigate to the `src/inference/` directory and run the inference script:

```
bash
python inference.py
```

### 📓 **Example Notebook**

Refer to `examples/example_usage.ipynb` for a comprehensive guide on setting up, training, and performing inference with AutoRLAIF.

## 📄 **License**

This project is licensed under the [Apache License 2.0](LICENSE).

## 🤝 **Contributing**

Contributions are welcome! Please read [Kaggle大模型竞赛技术报告.md](Kaggle大模型竞赛技术报告.md) for guidelines on how to proceed.

## 📧 **Contact**

For any questions or suggestions, please contact dkxuyangyang@163.com.

