FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy model (multilingual)
RUN python -m spacy download xx_ent_wiki_sm

# Copy project
COPY . .

# Install package in editable mode
RUN pip install -e . --no-deps

ENV PYTHONPATH=/workspace
ENV TOKENIZERS_PARALLELISM=false

CMD ["bash"]
