"""
Analytical insight generation with Chain-of-Thought prompting
and self-consistency sampling.
Ospan et al. (2024), Section III-E.

Self-consistency: m=5 independent completions at T=0.7,
majority-voted over structured JSON outputs.
"""

import json
from typing import List, Dict, Optional
from collections import Counter
from dataclasses import dataclass, field

from .clusterer import ThematicCluster


@dataclass
class Insight:
    cluster_id: int
    topic: str
    claims: List[Dict] = field(default_factory=list)
    raw_text: str = ""


STATIC_SYSTEM_PROMPT = """Сіз Қазақстан журналистикасы саласының сарапшысысыз.
Берілген мәтіннен кесте жасаңыз. Кесте міндетті түрде мына бағандардан тұруы керек:
Тақырып | Оқиға | Көрсеткіш | Кезең | Қосымша мәлімет

Ережелер:
1. Тек мәтінде нақты айтылған фактілерді жазыңыз
2. Мәліметі жоқ ұяшықтарға "-" белгісін қойыңыз
3. Кестені Markdown форматында (pipe-delimited) шығарыңыз
4. Тақырыптар: Энергетика, Ауылшаруашылық, Цифрлық трансформация, т.б."""

DYNAMIC_SYSTEM_PROMPT = """Сіз деректерді журналистика саласының сарапшысысыз.
Берілген мәтіннің мазмұнына сай оңтайлы кесте бағандарын өзіңіз анықтаңыз.

Нұсқаулар:
1. Мәтіннің мазмұнына қарай 2-8 баған таңдаңыз
2. Баған атауларын мәтіннің семантикасынан шығарыңыз
3. Тек нақты фактілерді жазыңыз (галлюцинация болмасын)
4. Кестені Markdown форматында шығарыңыз
5. Бос ұяшықтарға "-" қойыңыз"""

COT_USER_TEMPLATE = """Мына мәтінді талдап, кесте жасаңыз:

{text}

Қадамдар:
1. Негізгі тақырыпты анықтаңыз
2. Барлық сандық деректерді тізімдеңіз
3. Әрбір деректі бастапқы сөйлемге байланыстырыңыз
4. Кестені жасаңыз"""


class InsightGenerator:
    """
    Generates structured table insights from thematic clusters using
    LLM with CoT prompting and self-consistency aggregation.
    """

    def __init__(
        self,
        base_model: str,
        lora_adapter: Optional[str] = None,
        device: str = "auto",
        m: int = 5,
        temperature: float = 0.7,
        max_new_tokens: int = 1500,
    ):
        self.m = m
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.regime = "dynamic"  # set by pipeline
        self._model = None
        self._tokenizer = None
        self._load_model(base_model, lora_adapter, device)

    def _load_model(self, base_model: str, lora_adapter: Optional[str], device: str):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel

            self._tokenizer = AutoTokenizer.from_pretrained(base_model)
            self._model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            if lora_adapter:
                self._model = PeftModel.from_pretrained(self._model, lora_adapter)
                self._model = self._model.merge_and_unload()
        except ImportError:
            print("WARNING: transformers/peft not available. Using mock generation.")

    def generate(self, clusters: List[ThematicCluster]) -> List[Insight]:
        insights = []
        for cluster in clusters:
            insight = self._generate_cluster_insight(cluster)
            insights.append(insight)
        return insights

    def _generate_cluster_insight(self, cluster: ThematicCluster) -> Insight:
        text = cluster.text
        system_prompt = (DYNAMIC_SYSTEM_PROMPT
                         if self.regime == "dynamic"
                         else STATIC_SYSTEM_PROMPT)

        completions = []
        for _ in range(self.m):
            completion = self._call_model(text, system_prompt)
            completions.append(completion)

        # Self-consistency: majority vote over completions
        best = self._majority_vote(completions, text)

        return Insight(
            cluster_id=cluster.cluster_id,
            topic=f"cluster_{cluster.cluster_id}",
            raw_text=best,
        )

    def _call_model(self, text: str, system_prompt: str) -> str:
        if self._model is None:
            return self._mock_response(text)

        import torch
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": COT_USER_TEMPLATE.format(text=text)},
        ]
        input_ids = self._tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = output[0][input_ids.shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def _majority_vote(self, completions: List[str], source_text: str) -> str:
        """Return most common completion (or first if all unique)."""
        counts = Counter(completions)
        return counts.most_common(1)[0][0]

    def _mock_response(self, text: str) -> str:
        """Fallback for environments without GPU/model."""
        return "| Тақырып | Мәлімет |\n|---------|----------|\n| Мәтін | - |"
