# Fine-Tuning
📊 Notebook: Untitled58.ipynb
This notebook focuses on sentiment classification using both zero-shot and fine-tuned approaches with Hugging Face transformers. It is designed to test models on news-based text inputs, comparing predicted and actual sentiments.

🎯 Purpose
Perform zero-shot and supervised sentiment analysis on small-scale datasets across balanced categories (positive, negative, neutral).

Notebook Goal: “First do 50 to 60 (equally distributed between neutral, positive, and negative), zero-shot”

🔧 Libraries Used
transformers, torch, pandas, numpy

openai, langchain, TrainingArguments, AutoTokenizer

Trainer, pipeline, csv, json

🧠 Key Functions
read_sentences_from_file(): Loads input sentences

categorize_sentiment(): Assigns zero-shot classification labels

fine_tune_model(): Fine-tunes a transformer for 3-class sentiment analysis

evaluate_baseline_accuracy(): Compares model output to labeled sentiment

test_on_news_items(): Runs evaluations on realistic samples

write_results_to_csv(): Outputs model predictions for reporting

This notebook is useful for training/testing sentiment models on custom small datasets and comparing zero-shot vs fine-tuned accuracy—ideal for low-resource text classification tasks.
