# %% [markdown]
# # Part F – Training and Evaluation Challenges
#
# A short analysis discussing challenges in training and evaluating LLMs.

# %% [markdown]
# ## 1. Challenges in Training LLMs
#
# Training large language models presents several significant challenges:
#
# **Computational Cost**: Training even moderately-sized LLMs (7B+ parameters) requires
# hundreds of GPU-hours on high-end hardware (A100, H100). For example, LLaMA-2 70B
# required ~1.7 million GPU-hours on A100-80GB GPUs. Small research labs and universities
# often lack access to such resources, making training from scratch impractical.
#
# **Training Instability**: Loss spikes, gradient explosions, and divergence are common
# during LLM training. These issues often require manual intervention — rolling back to
# earlier checkpoints, adjusting learning rates, or modifying data mixtures mid-training.
#
# **Hyperparameter Sensitivity**: Learning rate schedules, warmup steps, batch sizes, and
# weight decay all significantly impact final model quality. Finding optimal settings
# often requires expensive hyperparameter sweeps.

# %% [markdown]
# ## 2. Data Requirements
#
# **Scale**: Modern LLMs are trained on trillions of tokens. GPT-3 used ~300B tokens,
# while LLaMA-2 used ~2T tokens. Collecting, cleaning, and deduplicating such massive
# datasets is a major engineering challenge.
#
# **Quality vs Quantity**: Data quality matters more than raw volume. Models trained on
# curated, high-quality data (e.g., textbooks, Wikipedia) often outperform those trained
# on larger but noisier web crawls. The "Textbooks Are All You Need" paper demonstrated
# this with the Phi series of models.
#
# **Domain Coverage**: For specialized tasks (agriculture, medicine, law), the model
# needs sufficient domain-specific data. General web data alone is insufficient for
# strong performance on niche topics, which is why fine-tuning on domain datasets
# (like the Agriculture-QA dataset we used in Part B) is necessary.
#
# **Bias and Toxicity**: Training data from the internet inherently contains biases,
# stereotypes, and toxic content. Filtering these while retaining useful data is an
# ongoing research challenge.

# %% [markdown]
# ## 3. Compute Limitations
#
# **VRAM Constraints**: A 7B parameter model in fp16 requires ~14GB of VRAM just for
# weights. During training, optimizer states (Adam uses 2x model size) and gradients
# push total memory to ~4x model size. This means training a 7B model needs ~56GB VRAM.
#
# **Solutions we used**:
# - **4-bit Quantization (QLoRA)**: Reduces model weight memory by ~4x, enabling
#   inference of 7B models on 8GB GPUs (as demonstrated in Part D)
# - **Gradient Accumulation**: Simulates larger batch sizes without extra VRAM
# - **Mixed Precision (fp16)**: Halves computation and memory during training
# - **Small Language Models**: Using DistilGPT-2 (82M params) instead of larger
#   models for fine-tuning (as in Part B) makes training feasible on consumer hardware
#
# **Training Time**: Even with GPU acceleration, fine-tuning takes hours. Training
# from scratch takes weeks to months. CPU-only training is impractical for any
# model larger than ~100M parameters.

# %% [markdown]
# ## 4. Evaluation Difficulties
#
# **No Single Metric**: Unlike classification (accuracy) or regression (MSE), there
# is no universally accepted metric for generative text quality. We used multiple
# metrics in Part C (Perplexity, BLEU, ROUGE, human evaluation) because each captures
# different aspects.
#
# **Metric Limitations**:
# - **Perplexity**: Measures how well the model predicts text, but low perplexity
#   does not guarantee useful or factual outputs
# - **BLEU/ROUGE**: Measure surface-level word overlap with references, but a
#   semantically correct answer phrased differently would score poorly
# - **Human Evaluation**: Most reliable but expensive, subjective, and not scalable
#
# **Benchmark Saturation**: Popular benchmarks (MMLU, HellaSwag, etc.) are increasingly
# saturated, with top models scoring near ceiling. This makes it hard to differentiate
# between models. Additionally, there are concerns about data contamination — models
# may have seen benchmark questions during training.
#
# **Task-Specific vs General**: A model may score well on benchmarks but perform
# poorly on real-world tasks, or vice versa. Domain-specific evaluation (like our
# agriculture Q&A evaluation) is essential.

# %% [markdown]
# ## 5. Hallucination Issues
#
# **Definition**: Hallucination occurs when LLMs generate text that sounds plausible
# but is factually incorrect, fabricated, or unsupported by the input context.
#
# **Types of Hallucination**:
# - **Factual hallucination**: Generating incorrect facts (e.g., wrong crop growth
#   cycles, incorrect pesticide recommendations)
# - **Faithfulness hallucination**: In RAG systems (like Part E), generating answers
#   that are not supported by the retrieved documents
# - **Fabrication**: Inventing citations, statistics, or sources that do not exist
#
# **Why It Happens**:
# - Models learn statistical patterns, not factual knowledge
# - Training data may contain contradictory information
# - Models are trained to generate fluent text, not necessarily truthful text
# - Smaller models (like the ones we used) are more prone to hallucination
#
# **Mitigation Strategies**:
# - **RAG (Retrieval-Augmented Generation)**: As implemented in Part E, grounding
#   responses in retrieved documents reduces hallucination
# - **Fine-tuning on domain data**: As done in Part B, improves factual accuracy
#   within the specific domain
# - **Temperature control**: Lower temperatures produce more deterministic outputs
# - **Output verification**: Post-processing to check facts against knowledge bases
# - **RLHF**: Reinforcement Learning from Human Feedback teaches models to be more
#   truthful, though it requires significant resources

# %% [markdown]
# ## Summary Table
#
# | Challenge | Impact | Our Mitigation |
# |---|---|---|
# | Compute cost | Cannot train large models | Used DistilGPT-2 (82M) + 4-bit quantization |
# | Data requirements | Need domain-specific data | Agriculture-QA dataset from HuggingFace |
# | Training instability | Loss spikes, divergence | Small model + low learning rate + warmup |
# | Evaluation difficulty | No single reliable metric | Multiple metrics (PPL, BLEU, ROUGE, human) |
# | Hallucination | Incorrect/fabricated outputs | RAG pipeline (Part E) + domain fine-tuning |

# %%
print("Part F: Training and Evaluation Challenges - Analysis Complete")
print("See the markdown cells above for the full discussion.")
