import pandas as pd
from tqdm import tqdm
from rouge import Rouge
from bert_score import BERTScorer
from IPython.display import (
    display, 
    HTML, 
    Markdown
)


class Metric_Evaluation:

    def __init__(self):
        self.rouge = Rouge()
        self.scorer = BERTScorer(lang="en")


    def eval_bert_and_rouge_summary(self, gen_gt_responses, average=True):

        rouge_metrics = {
            "rouge-1 (Precision)": [], 
            "rouge-1 (Recall)": [],
            "rouge-1 (F-Score)": [],
            "rouge-2 (Precision)": [],
            "rouge-2 (Recall)": [],
            "rouge-2 (F-Score)": [],
            "rouge-l (Precision)": [],
            "rouge-l (Recall)": [],
            "rouge-l (F-Score)": []
        }
        for (generated_summary, gt_summary) in tqdm(gen_gt_responses, total=len(gen_gt_responses)):
            eval_rouge_score = self.rouge.get_scores(generated_summary, gt_summary)
            
            for metric in ["rouge-1", "rouge-2", "rouge-l"]:
                for label in ["Precision", "Recall", "F-Score"]:
                    eval_score = eval_rouge_score[0][metric][label[0].lower()]
                    rouge_metrics[f"{metric} ({label})"].append(eval_score)

        # Calculate BERTScore for the summary 1 against the excerpt
        generated_summaries = [generated_summary for (generated_summary, _) in gen_gt_responses]
        gt_summaries = [gt_summary for (_, gt_summary) in gen_gt_responses]
        prec_bert, recall_bert, f1_bert = self.scorer.score(generated_summaries, gt_summaries)
        bert_mertics = {"Precision (bert)": prec_bert, "Recall (bert)": recall_bert, "F1 (bert)": f1_bert}

        df_metrics = pd.concat([pd.DataFrame(rouge_metrics), pd.DataFrame(bert_mertics)], axis=1)

        if average:
            df_metrics_raw = df_metrics.copy(deep=True)
            df_metrics = (
                df_metrics.mean().to_frame().rename(columns={0: "Metric Value"})
                .style.background_gradient(cmap='YlGn', axis=None)
            )
            return df_metrics, df_metrics_raw

        return df_metrics, None


def pretty_print(text_a, text_b=None):

    text_a = text_a.replace('$', '\$')
    if text_b is not None:
        text_b = text_b.replace('$', '\$')

    title_a, title_b = None, None

    if text_b is not None:
        title_a = "Model Response"
        title_b = "Ground Truth"
    else:
        title_a = "Model Response"
    
    html_code_a = f"""
    <div style="display: flex; justify-content: space-between; gap: 20px;">
        <div style="flex: 1; border: 1px solid #ddd; padding: 20px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); border-radius: 8px; overflow-wrap: break-word; background-color: #efdfdf;">
            <h3 style="font-family: Arial, sans-serif; font-weight: bold; color: #333;">{title_a}</h3>
            <p style="font-family: Arial, sans-serif; color: #555; line-height: 1.6;">{text_a}</p>
        </div>
    """

    if text_b is not None:
        html_code_b = f"""
            <div style="flex: 1; border: 1px solid #ddd; padding: 20px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); border-radius: 8px; overflow-wrap: break-word; background-color: #eef7f9;">
                <h3 style="font-family: Arial, sans-serif; font-weight: bold; color: #333;">{title_b}</h3>
                <p style="font-family: Arial, sans-serif; color: #555; line-height: 1.6;">{text_b}</p>
            </div>
        """
        html_code_a += html_code_b

    html_code = html_code_a + "</div>"
    
    # Display the HTML code in the notebook
    display(HTML(html_code))
    