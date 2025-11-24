A comprehensive machine learning project that estimates product prices from textual descriptions using a progression of techniques: from traditional ML baselines and frontier LLMs to advanced Parameter-Efficient Fine-Tuning (PEFT) of open-source models.

📊 Model Performance Comparison
Model	Type	Training Data	Avg Error ($)	RMSLE	Green Hits (%)	Cost per 250 items
Random Pricer	Baseline	None	~300-400	Highest	~10-15%	Free
Constant Pricer	Baseline	None	~60-80	High	~20-25%	Free
Linear Regression	Traditional ML	400k items	~50-70	Moderate	~30-40%	Free
Bag-of-Words + LR	Traditional ML	400k items	~40-60	Moderate	~35-45%	Free
Word2Vec + LR	Traditional ML	400k items	~35-50	Moderate-Low	~40-50%	Free
Linear SVR	Traditional ML	400k items	~35-50	Moderate-Low	~40-50%	Free
Random Forest	Traditional ML	400k items	~30-45	Low	~45-55%	Free
Human Baseline	Manual	None	~50-100	Moderate-High	~30-40%	Free
GPT-4o-mini (zero-shot)	Frontier LLM	Pretrained	~20-35	Low	~60-70%	~$0.01-0.02
GPT-4o (zero-shot)	Frontier LLM	Pretrained	~15-30	Very Low	~65-75%	~$0.01-0.02
Claude 3.5 Sonnet (zero-shot)	Frontier LLM	Pretrained	~20-35	Low	~60-70%	~$0.01-0.02
GPT-4o-mini (fine-tuned)	Fine-tuned LLM (Closed)	200 examples	~15-25	Very Low	~70-80%	~$0.50-1.00
Llama-3.1-8B (PEFT/LoRA)	Fine-tuned LLM (Open)	200 examples	~10-15	Ultra Low	~80-85%	Low (GPU cost)

Green hits = predictions within $40 or 20% error threshold. Performance metrics are approximate and may vary based on test set composition.

📚 Project Overview
Day 1: Data Exploration & Curation Foundation
Concept: Established the data pipeline foundation by exploring Amazon product metadata and implementing the Item class for data cleaning and prompt generation. The Item class uses the Llama 3.1 tokenizer to ensure all product descriptions fit within a 180-token limit, creating standardized training prompts. We analyzed price distributions, token counts, and validated that our cleaning pipeline (removing SKUs, normalizing text) produces high-quality training examples. The notebook loads appliance data from Hugging Face, filters items with valid prices ($1-999), and generates prompts in the format "How much does this cost? [description] Price is $X.00" for supervised learning.

What We Did: Loaded Amazon Reviews 2023 appliance dataset from Hugging Face, explored raw data structure (title, description, features, details, price), analyzed price and length distributions with histograms, implemented Item class for text scrubbing and tokenization, validated prompt generation with 150-180 token constraints, and created initial curated dataset of appliance items ready for training.

Day 2: Multi-Category Dataset Curation
Concept: Scaled the curation pipeline to multiple product categories (Automotive, Electronics, Office Products, Tools, Cell Phones, Toys, Appliances, Musical Instruments) to build a diverse 400k-item training set. Implemented intelligent price balancing using weighted sampling to reduce bias toward cheap items and balance category representation. The curation process filters items by price range ($0.50-$999.49), applies token length constraints, and creates balanced train/test splits. We saved curated datasets as pickle files for efficient loading and created histograms to visualize price/token distributions across categories.

What We Did: Extended ItemLoader class to process multiple categories in parallel, loaded 8 product categories from Hugging Face datasets, applied price balancing algorithm (keeping all items $240+, weighted sampling for cheaper items to reduce Automotive bias), created 400k training / 2k test split with proper shuffling, saved datasets as train.pkl and test.pkl for future use, and generated visualization charts showing balanced price distributions and category representation.

Day 3: Traditional Machine Learning Baselines
Concept: Implemented and evaluated seven traditional ML models as baselines, ranging from trivial (random guess, constant average) to sophisticated (Random Forest with Word2Vec embeddings). Each model processes product descriptions differently: hand-crafted features (weight, rank, text length), bag-of-words representations, dense embeddings via Word2Vec, and ensemble methods. The Tester class provides standardized evaluation metrics (average error, RMSLE, color-coded hit rates) to compare model performance. These baselines establish a performance floor that frontier models and fine-tuning must beat.

What We Did: Loaded curated train/test pickle files, implemented seven baseline models (random, constant, linear regression with features, bag-of-words LR, Word2Vec+LR, Linear SVR, Random Forest), extracted features from product details JSON (weight, best-seller rank, brand), trained Word2Vec embeddings on product descriptions, evaluated all models using shared Tester harness with 250 test items, and visualized results showing Random Forest as best traditional ML performer with ~30-45% average error.

Day 4: Frontier Model Evaluation
Concept: Tested state-of-the-art LLMs (GPT-4o-mini, GPT-4o, Claude 3.5 Sonnet) in zero-shot mode without fine-tuning to compare against traditional ML baselines. These models leverage massive pretraining on diverse text, potentially including Amazon product data, giving them inherent price estimation capabilities. We formatted prompts as chat messages (system instruction + user product description) and used get_price() regex utility to extract numeric prices from model responses. Also included human baseline comparison via CSV files to understand human performance. Frontier models significantly outperformed traditional ML, demonstrating the power of large-scale pretraining.

What We Did: Set up OpenAI and Anthropic API clients, created messages_for() function to format prompts as chat conversations, implemented get_price() utility to extract prices from model responses, tested GPT-4o-mini, GPT-4o, and Claude 3.5 Sonnet on 250 test items, generated human_input.csv/human_output.csv for manual evaluation, and compared all models showing frontier LLMs achieving ~15-35% average error vs ~30-45% for traditional ML, with GPT-4o performing best at ~15-30% error.

Day 5: Fine-Tuning GPT-4o-mini
Concept: Fine-tuned GPT-4o-mini on 200 curated training examples using OpenAI's fine-tuning API to create a specialized price estimation model. Converted Item prompts to JSONL format (JSON Lines) with chat message structure required by OpenAI, split data into 200 training / 50 validation examples, and uploaded files to OpenAI. The fine-tuning process adapts the pretrained model to our specific task, learning price patterns from product descriptions. Fine-tuned models typically outperform zero-shot frontier models on domain-specific tasks. Optional Weights & Biases integration provides training metrics visualization.

What We Did: Selected 200 training examples from curated dataset, created make_jsonl() and write_jsonl() functions to convert prompts to OpenAI's JSONL format, uploaded training/validation files via OpenAI API, created fine-tuning job with GPT-4o-mini base model (1 epoch, seed=42), monitored job status through validation → queued → running → succeeded states, tested fine-tuned model using same evaluation harness, and achieved best performance (~15-25% average error, ~70-80% green hits) compared to all previous models including zero-shot frontier models.

PEFT Fine-Tuning of Open-Source LLMs
Concept: The final phase of the project involved fine-tuning a powerful open-source Large Language Model (e.g., Llama 3.1 8B) using Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically LoRA (Low-Rank Adaptation). This approach allows us to achieve state-of-the-art performance by training only a small fraction of the model's parameters, making the process computationally feasible on consumer-grade GPUs. We then loaded the base model and the trained LoRA adapters for efficient inference. This method proved superior to all other approaches, including fine-tuning closed-source frontier models.

What We Did:

Selected an open-source base model (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct).

Configured LoRA parameters (rank, alpha, dropout) for efficient training.

Fine-tuned the model on our curated training dataset using the peft and transformers libraries.

Implemented an inference pipeline that loads the base model and the trained LoRA adapters.

Evaluated the PEFT-tuned model on the test set, demonstrating superior performance with the lowest average error and highest "green hit" rate of all models tested.

🛠️ Environment Setup
Create virtual environment: python3.11 -m venv .venv then source .venv/bin/activate

Install dependencies: pip install -r requirements.txt (ensure peft, transformers, accelerate, and bitsandbytes are included)

Configure secrets: Create .env file with:

HF_TOKEN=hf_...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
WANDB_API_KEY=... (optional)
Run notebooks: Execute the notebooks sequentially, ending with the PEFT fine-tuning and inference notebook.

📦 Key Artifacts
train.pkl, test.pkl - Full multi-category datasets (400k train / 2k test)

train_lite.pkl, test_lite.pkl - Appliance-only datasets for faster experimentation

fine_tune_train.jsonl, fine_tune_validation.jsonl - JSONL files for OpenAI fine-tuning

peft_adapters/ - Directory containing the fine-tuned LoRA adapters for the open-source model.

human_input.csv, human_output.csv - Manual evaluation data

items.py, loaders.py, testing.py - Core Python modules for data processing and evaluation

🔑 APIs & Tokens Required
HF_TOKEN (required): Hugging Face token for dataset access and downloading gated models like Llama 3.1.

OPENAI_API_KEY (Day 4+): For frontier model testing and fine-tuning.

ANTHROPIC_API_KEY (Day 4): For Claude model testing.

WANDB_API_KEY (Day 5, optional): For training metrics visualization.

⚠️ Common Issues & Solutions
GPU Memory Errors: PEFT fine-tuning, while efficient, still requires significant GPU memory. Ensure you are using a GPU with sufficient VRAM (e.g., A10G, A100) or adjust batch sizes and model quantization (e.g., using 4-bit quantization with bitsandbytes).

401/403 Hugging Face errors: Ensure HF_TOKEN is valid and has read access; request access to gated models (Meta Llama 3.1) at huggingface.co.

Long dataset load times: First run downloads ~1 hour; subsequent runs use cached data.

PyTorch warnings: Install torch==2.4.1 for local model inference.

OpenAI SDK httpx error: Pin httpx<0.28 in requirements.txt (already included).

W&B sync blocking: Use wait_for_job_success=False or skip W&B entirely for faster workflow.

GitHub push protection: Ensure .env is in .gitignore and removed from git history before pushing.

🎯 Key Concepts Explained
PEFT (Parameter-Efficient Fine-Tuning): A set of techniques that allow fine-tuning large language models by adapting only a small subset of parameters, significantly reducing computational and storage costs. We used LoRA (Low-Rank Adaptation) for this project.

Tokenization: Product descriptions are converted to tokens (subword pieces) using Llama's tokenizer. We constrain to 160 tokens for descriptions, adding question/answer text brings total to ~180 tokens per training example.

Prompt Engineering: Training prompts follow format "How much does this cost? [cleaned description] Price is $X.00" - this teaches the model to complete the price given the description.

Zero-Shot vs Fine-Tuning: Zero-shot models use pretrained knowledge without task-specific training. Fine-tuning adapts pretrained models to our specific dataset, typically improving performance.

Evaluation Metrics: Average error (absolute dollar difference), RMSLE (Root Mean Squared Log Error for relative accuracy), and green hits (predictions within $40 or 20% of true price).

JSONL Format: JSON Lines format required by OpenAI - each line is a JSON object with "messages" array containing system/user/assistant roles for chat-based fine-tuning.

🚀 Next Steps
Deploy the fine-tuned PEFT model using a serving framework like Text Generation Inference (TGI) or vLLM.

Experiment with different open-source base models (e.g., Mistral, Gemma) to compare performance.

Build a user-friendly frontend application for real-time price estimation.

Implement a continuous retraining pipeline to keep the model updated with new product data.
