# app/main.py
from flask import Flask, request, jsonify, render_template
import time
import threading # 非同期処理のため

# プロジェクトルートからの相対パスでモジュールをインポート
from app.data_collector import DataCollector
from app.preprocessor import Preprocessor
from app.analyzer import SentimentAnalyzer
from app.reporter import ReportGenerator
# pandasをインポート (reporter.pyで使っているため)
import pandas as pd 

app = Flask(__name__)

# 各モジュールのインスタンス化 (アプリケーション起動時に一度だけ行う)
# これらはスレッドセーフではない場合があるので、リクエスト毎に生成するか、
# スレッドセーフな設計にするか、Gunicorn等のWSGIサーバーでプロセス数を調整する
# ここでは簡略化のためグローバルに持つが、大規模アプリでは注意
data_collector = DataCollector()
preprocessor = Preprocessor()
sentiment_analyzer = SentimentAnalyzer() # モデルのロードに時間がかかる
report_generator = ReportGenerator()

# 処理状況を格納する辞書 (デモ用、本番ではDBやキャッシュストアを使う)
analysis_status = {}
analysis_results = {}

def process_analysis(task_id, keyword):
    """実際の分析処理を行う関数 (バックグラウンドスレッドで実行)"""
    analysis_status[task_id] = "processing"
    try:
        print(f"Task {task_id}: Starting analysis for '{keyword}'")
        # 1. データ収集
        raw_tweets = data_collector.search_tweets(keyword, max_results=50) # max_resultsは調整
        if not raw_tweets:
            analysis_results[task_id] = "該当するツイートは見つかりませんでした。"
            analysis_status[task_id] = "completed"
            return
        
        # 2. 前処理
        processed_tweets_for_analysis = []
        cleaned_texts_for_sentiment = []
        for tweet_data in raw_tweets:
            processed_data = preprocessor.preprocess_tweet(tweet_data)
            if processed_data.get('cleaned_text'):
                cleaned_texts_for_sentiment.append(processed_data['cleaned_text'])
            processed_tweets_for_analysis.append(processed_data) # 元のデータも保持

        # 感情分析 (バッチ処理)
        sentiment_results = []
        if cleaned_texts_for_sentiment:
            # pipelineにテキストのリストを渡す
            # モデルやpipelineの実装によっては、バッチサイズを指定できる場合がある
            # (例: sentiment_pipeline(cleaned_texts_for_sentiment, batch_size=8))
            sentiment_pipeline_results = sentiment_analyzer.sentiment_pipeline(
                cleaned_texts_for_sentiment, truncation=True, max_length=512
            )
            # sentiment_pipeline_results の形式を確認して、元のツイートデータと結合する
            # sentiment_analyzer.analyze_sentiment を少し変更して、リストを受け付けるようにするのも手
            # ここでは簡易的に、1件ずつ呼び出す形に戻すが、上記を参考に改良できる
            idx = 0
            for p_tweet in processed_tweets_for_analysis:
                if p_tweet.get('cleaned_text'):
                    # sentiment_pipeline_results から対応する結果を取得
                    # analyze_sentiment を少し改修して、pipelineの直接結果を整形する形にしても良い
                    current_sentiment_result = sentiment_analyzer.map_pipeline_result(sentiment_pipeline_results[idx])
                    p_tweet['sentiment'] = current_sentiment_result
                    idx += 1
                else:
                    p_tweet['sentiment'] = {'label': 'ニュートラル', 'score': 0.0, 'note': 'Empty after cleaning'}
        analyzed_tweets = processed_tweets_for_analysis

        

        # 3. レポート生成
        report = report_generator.generate_report(keyword, analyzed_tweets)
        analysis_results[task_id] = report
        analysis_status[task_id] = "completed"
        print(f"Task {task_id}: Analysis completed for '{keyword}'")

    except Exception as e:
        print(f"Task {task_id}: Error during analysis for '{keyword}': {e}")
        analysis_results[task_id] = f"分析中にエラーが発生しました: {e}"
        analysis_status[task_id] = "failed"


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html') # 簡単な入力フォームを表示

@app.route('/analyze', methods=['POST'])
def analyze_keyword_async():
    data = request.json
    keyword = data.get('keyword')
    if not keyword:
        return jsonify({"error": "キーワードが指定されていません"}), 400

    task_id = str(int(time.time() * 1000)) # ユニークなタスクIDを生成
    analysis_status[task_id] = "pending"
    analysis_results[task_id] = None

    # バックグラウンドスレッドで重い処理を実行
    thread = threading.Thread(target=process_analysis, args=(task_id, keyword))
    thread.start()
    
    return jsonify({"message": "分析リクエストを受け付けました。", "task_id": task_id}), 202


@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    status = analysis_status.get(task_id)
    if not status:
        return jsonify({"error": "タスクが見つかりません"}), 404
    
    result = None
    if status == "completed" or status == "failed":
        result = analysis_results.get(task_id)
        # メモリリークを防ぐため、取得後は削除する（本番ではDBなどで永続化）
        # del analysis_status[task_id]
        # del analysis_results[task_id]

    return jsonify({"task_id": task_id, "status": status, "result": result})


if __name__ == '__main__':
    # 開発用サーバーの起動。本番環境ではGunicornなどを使用
    app.run(debug=True, host='0.0.0.0', port=5001) # portは適宜変更