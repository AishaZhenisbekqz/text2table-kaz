#!/usr/bin/env bash
# CMES-86051: Разделы 1-4 инструкции — безопасность, клонирование репозитория,
# фиксация окружения и model revisions.
#
# Запускать на машине с RTX 5090. Перед запуском:
#   export HF_TOKEN="..."          # новый read-only токен (старый уже отозван)
#   export OPENAI_API_KEY="..."    # понадобится позже, для ноутбука 01
#
# Использование:
#   bash 00_setup_environment.sh /path/to/CMES_86051_Reproducible_Evaluation_Package_27_07_26.zip

set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Использование: bash 00_setup_environment.sh /path/to/package.zip"
  exit 1
fi
PACKAGE_ZIP="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
if [ ! -f "$PACKAGE_ZIP" ]; then
  echo "ОШИБКА: файл не найден: $PACKAGE_ZIP"
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ОШИБКА: переменная среды HF_TOKEN не установлена. Не продолжаю."
  echo "  export HF_TOKEN=\"...\""
  exit 1
fi

WORKDIR="$(pwd)/cmes_86051_run_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "Рабочая папка: $WORKDIR"

# --- лог всех команд с этого момента ---
exec > >(tee -a run_commands.log) 2>&1
echo "=== run_commands.log начат: $(date -u -Iseconds) ==="

# ============================================================
# Раздел 2: проверка и фиксация исходных версий
# ============================================================

echo ""
echo "### Клонирование репозитория ###"
git clone https://github.com/AishaZhenisbekqz/text2table-kaz.git
cd text2table-kaz

REQUIRED_COMMIT="ee537aaf1d3395b69f43f85998cbf0c8738aa636"
git checkout "$REQUIRED_COMMIT"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
echo "$ACTUAL_COMMIT" | tee ../repo_commit.txt

if [ "$ACTUAL_COMMIT" != "$REQUIRED_COMMIT" ]; then
  echo "ОШИБКА: checkout не на нужном commit. Ожидался $REQUIRED_COMMIT, получен $ACTUAL_COMMIT"
  exit 1
fi
cd ..

echo ""
echo "### Распаковка пакета эксперимента ###"
mkdir -p CMES_evaluation
unzip -o "$PACKAGE_ZIP" -d CMES_evaluation

echo ""
echo "### Проверка контрольных хешей и числа записей ###"
EXPECTED_PACKAGE_SHA="0c40a26476e0de4c2d4af3e3ba83eab1c16bf6c1478f846fd26a66d6d329f75"
EXPECTED_GOLD_SHA="95eb0fd2fe1a62fea911379e79a357722e41ca73b1b7e2431beda219096061b4"
EXPECTED_GOLD_LINES=1000

ACTUAL_PACKAGE_SHA="$(sha256sum "$PACKAGE_ZIP" | awk '{print $1}')"
ACTUAL_GOLD_SHA="$(sha256sum CMES_evaluation/benchmark_gold_1000.jsonl | awk '{print $1}')"
ACTUAL_GOLD_LINES="$(wc -l < CMES_evaluation/benchmark_gold_1000.jsonl | tr -d ' ')"

echo "package sha256:  ожидается=$EXPECTED_PACKAGE_SHA  фактически=$ACTUAL_PACKAGE_SHA"
echo "gold sha256:     ожидается=$EXPECTED_GOLD_SHA      фактически=$ACTUAL_GOLD_SHA"
echo "gold lines:      ожидается=$EXPECTED_GOLD_LINES    фактически=$ACTUAL_GOLD_LINES"

# ПРИМЕЧАНИЕ: в инструкции указан хэш пакета из 64 hex-символов; сверьте его
# самостоятельно с тем, что реально прислал автор -- значение выше взято из
# текста инструкции as-is и может отличаться посимвольно, проверьте вручную.
if [ "$ACTUAL_GOLD_SHA" != "$EXPECTED_GOLD_SHA" ] || [ "$ACTUAL_GOLD_LINES" != "$EXPECTED_GOLD_LINES" ]; then
  echo "ОСТАНОВКА: hash или число строк Gold-файла не совпадают с ожидаемыми."
  echo "Полный эксперимент запускать нельзя до выяснения причины (см. Раздел 2 инструкции)."
  exit 1
fi
echo "Gold-бенчмарк подтверждён: 1000 записей, хеш совпадает."

# ============================================================
# Раздел 3: Python-окружение
# ============================================================

echo ""
echo "### Создание venv и установка зависимостей ###"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r text2table-kaz/Text2Table_Kaz/requirements.txt
python -m pip install nbformat jupyter huggingface_hub openai tqdm  # для ноутбуков 01-03
python -m pip freeze > environment_pip_freeze.txt
python --version > python_version.txt
nvidia-smi > nvidia_smi.txt

echo "Если pip install упал из-за несовместимости версий с CUDA/моделью -- НЕ обновлять"
echo "молча. Сохранить текст ошибки, подобрать реально совместимые версии, повторить"
echo "smoke test, и зафиксировать фактические версии в environment_pip_freeze.txt и в статье."

# ============================================================
# Раздел 4: точные ревизии моделей на Hugging Face
# ============================================================

echo ""
echo "### Получение HF commit SHA обеих моделей ###"
python - "$HF_TOKEN" <<'PYEOF'
import json
import sys
from huggingface_hub import model_info

token = sys.argv[1]
model_ids = [
    "AishaSailau/qwen3.5-text2table-static",
    "AishaSailau/qwen3.5-text2table-dynamic",
]

revisions = {}
for model_id in model_ids:
    info = model_info(model_id, token=token)
    revisions[model_id] = info.sha
    print(model_id, info.sha)

with open("model_revisions.json", "w", encoding="utf-8") as fh:
    json.dump(revisions, fh, ensure_ascii=False, indent=2)
PYEOF

echo ""
echo "=== Setup завершён: $(date -u -Iseconds) ==="
echo "Проверьте перед следующим шагом (smoke test generate_gold_predictions.py):"
echo "  - repo_commit.txt              -> должен совпадать с ee537aaf1d3395b69f43f85998cbf0c8738aa636"
echo "  - model_revisions.json         -> два реальных SHA, не пустые"
echo "  - environment_pip_freeze.txt   -> сверить с requirements.txt, зафиксировать расхождения"
echo "  - nvidia_smi.txt                -> подтвердить GPU >= 24GB"
echo "  - run_commands.log              -> полная история команд этого запуска"
