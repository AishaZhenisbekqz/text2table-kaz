"""
Dataset loading and prompt formatting for dual-regime training.
Ospan et al. (2024), Section IV-B.

Label masking (Equation 8):
    labels = [-100, ..., -100, y_1, ..., y_Na]
    Gradients flow only through table generation tokens.
"""

import json
from typing import Dict, List, Optional
from torch.utils.data import Dataset

# Fully Kazakh system prompts (native-speaker validated)
STATIC_SYSTEM = (
    "Сіз деректерді журналистика саласының сарапшысысыз. "
    "Берілген мәтіннен кесте жасаңыз. "
    "Кесте міндетті түрде мына бағандардан тұруы керек: "
    "Тақырып | Оқиға | Көрсеткіш | Кезең | Қосымша мәлімет. "
    "Тек нақты фактілерді жазыңыз. Мәліметі жоқ ұяшықтарға '-' қойыңыз. "
    "Кестені Markdown форматында шығарыңыз."
)

DYNAMIC_SYSTEM = (
    "Сіз деректерді журналистика саласының сарапшысысыз. "
    "Берілген мәтіннің мазмұнына сай оңтайлы кесте бағандарын өзіңіз анықтаңыз. "
    "2-8 баған таңдаңыз, баған атауларын мәтіннің семантикасынан шығарыңыз. "
    "Тек нақты фактілерді жазыңыз (галлюцинация болмасын). "
    "Кестені Markdown форматында шығарыңыз. Бос ұяшықтарға '-' қойыңыз."
)

USER_TEMPLATE = "Мына мәтіннен кесте жасаңыз:\n\n{text}"


class Text2TableDataset(Dataset):
    """
    Dataset for dual-regime text-to-table fine-tuning.

    Each instance: {"text": str, "table": str, "regime": str}
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        regime: str = "dynamic",
        max_context_length: int = 3000,
        max_generation_length: int = 1500,
    ):
        self.tokenizer = tokenizer
        self.regime = regime
        self.max_context_length = max_context_length
        self.max_generation_length = max_generation_length
        self.system_prompt = STATIC_SYSTEM if regime == "static" else DYNAMIC_SYSTEM
        self.data = self._load(data_path)

    def _load(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Filter by regime
        return [d for d in data if d.get("regime", self.regime) == self.regime]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        text = item["text"]
        table = item["table"]
        return self._format(text, table)

    def _format(self, text: str, table: str) -> Dict:
        """
        Format as chat template with label masking.
        Prompt tokens → -100, table tokens → actual ids.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": USER_TEMPLATE.format(text=text)},
            {"role": "assistant", "content": table},
        ]

        # Full sequence
        full = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )[0]

        # Prompt-only (to find where answer starts)
        prompt_messages = messages[:-1]
        prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )[0]

        # Apply label masking (Equation 8)
        labels = full.clone()
        labels[:len(prompt)] = -100

        # Truncate to max length
        max_len = self.max_context_length + self.max_generation_length
        if len(full) > max_len:
            full = full[:max_len]
            labels = labels[:max_len]

        return {
            "input_ids": full,
            "attention_mask": (full != self.tokenizer.pad_token_id).long(),
            "labels": labels,
        }
