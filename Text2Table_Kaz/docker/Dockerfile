FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y git curl wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download xx_ent_wiki_sm

COPY . /workspace/Text2Table_Kaz
WORKDIR /workspace/Text2Table_Kaz

ENV PYTHONPATH=/workspace/Text2Table_Kaz
ENV TOKENIZERS_PARALLELISM=false

CMD ["python", "scripts/train.py", "--config", "configs/dynamic_regime.yaml"]
