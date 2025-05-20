# app/analyzer.py
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging # ロギングを追加

logger = logging.getLogger(__name__)

# 使用するHugging Faceモデル名
MODEL_NAME = "Mizuiro-inc/bert-japanese-sentiment-analysis-large"

class SentimentAnalyzer:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.sentiment_pipeline = None # 初期化はメソッドで行う

        try:
            logger.info(f"Loading tokenizer for {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info(f"Loading model for {self.model_name}...")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # GPUが利用可能ならGPUを使う設定
            self.device = 0 if torch.cuda.is_available() else -1 
            # pipelineを初期化
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                # モデルによっては return_all_scores=True を設定すると全ラベルのスコアが返る
                # koheiduckモデルはデフォルトで最もスコアの高いラベルのみ
            )
            if self.device == 0:
                logger.info(f"Sentiment model '{self.model_name}' loaded successfully on GPU.")
            else:
                logger.info(f"Sentiment model '{self.model_name}' loaded successfully on CPU.")

        except Exception as e:
            logger.error(f"Error loading sentiment model '{self.model_name}': {e}", exc_info=True)
            # sentiment_pipeline は None のままになる


    def analyze_sentiment(self, text: str):
        if not self.sentiment_pipeline:
            logger.error("Sentiment pipeline is not initialized. Cannot analyze.")
            return {"label": "ERROR", "score": 0.0, "error_message": "Model not loaded"}

        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            logger.warning("Input text for sentiment analysis is empty or invalid.")
            return {"label": "NEUTRAL", "score": 0.0, "note": "Empty or invalid input text"}

        try:
            # モデルが処理できる最大トークン長を取得 (なければデフォルト512)
            # koheiduck/bert-japanese-finetuned-sentiment は BERT ベースなので 512 が一般的
            max_len = self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') else 512
            
            # truncation=True を指定すると、長すぎるテキストは自動的に切り詰めてくれる
            results = self.sentiment_pipeline(text, truncation=True, max_length=max_len)
            
            # pipelineの出力は通常リスト (要素数1のことが多い)
            # 例: [{'label': 'POSITIVE', 'score': 0.99...}]
            if results and isinstance(results, list) and len(results) > 0:
                result = results[0]
                # ラベル名を日本語にマッピング (モデルの出力ラベルに合わせて調整)
                # koheiduck/bert-japanese-finetuned-sentiment の出力は 'POSITIVE' or 'NEGATIVE'
                label_map = {
                    "POSITIVE": "ポジティブ",
                    "NEGATIVE": "ネガティブ",
                    "NEUTRAL": "ニュートラル",
                }
                
                final_label = label_map.get(result["label"].upper(), result["label"]) # 大文字に変換して検索
                
                return {
                    "label": final_label,
                    "score": round(result["score"], 4) # スコアを小数点以下4桁に丸める
                }
            else:
                logger.warning(f"Unexpected output format from sentiment pipeline for text: {text[:50]}... Output: {results}")
                return {"label": "UNKNOWN", "score": 0.0, "raw_output": results}

        except Exception as e:
            logger.error(f"Error during sentiment analysis for text '{text[:50]}...': {e}", exc_info=True)
            return {"label": "ERROR", "score": 0.0, "error_message": str(e)}

# テスト用 (main.pyから呼ばれるので、ここでの直接実行は必須ではない)
if __name__ == '__main__':
    # ロガーの基本設定 (テスト実行時のみ)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    analyzer = SentimentAnalyzer()

    if analyzer.sentiment_pipeline:
        test_texts = [
            "この映画、本当に感動した！素晴らしいストーリーだった。", # ポジティブ
            "今日のランチは最悪だった。味がひどいし、サービスも悪い。", # ネガティブ
            "まあ、悪くはないけど、期待していたほどではなかったな。", # ややネガティブ寄りか、モデル次第
            "特に何も感じなかった。", # ニュートラルに近いが、このモデルは2値分類
            "", # 空文字
            "これはペンです。", # 感情とは無関係な文
            "😊🎉🥳", # 絵文字のみ (モデルの学習データに依存)
            "長文テスト。この文章は非常に長く、モデルの最大入力長を超える可能性があります。その場合、適切に切り捨てられるか、あるいはエラーが発生するかを確認する必要があります。BERTベースのモデルでは通常512トークンが上限ですが、これは設定やモデルの実装によって異なる場合があります。適切な前処理とエラーハンドリングが重要です。このテスト文は、その挙動を確認するために意図的に長くしています。"
        ]

        for text_input in test_texts:
            sentiment = analyzer.analyze_sentiment(text_input)
            print(f"Text: {text_input[:70]}...")
            print(f"Sentiment: {sentiment}\n")
    else:
        print("Sentiment analyzer could not be initialized.")