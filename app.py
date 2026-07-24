import os
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

print("Loading AI Model...")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=os.environ.get("HF_TOKEN")
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=os.environ.get("HF_TOKEN")
)

print("Model Loaded Successfully!")