from typing import List, Dict
import evaluate

bleu_metric = evaluate.load("sacrebleu")
rouge_metric = evaluate.load("rouge")

def compute_text_metrics(preds: List[str], labels: List[str]) -> Dict[str, float]:
    """
    Compute BLEU and ROUGE-L between prediction strings and reference strings.
    """
    # sacrebleu expects list of hypotheses and list of list-of-references
    bleu = bleu_metric.compute(predictions=preds, references=[[l] for l in labels])
    rouge = rouge_metric.compute(predictions=preds, references=labels)

    return {
        "bleu": bleu["score"],
        "rougeL": rouge["rougeL"]
    }
