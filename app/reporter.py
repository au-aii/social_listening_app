import matplotlib
matplotlib.use('Agg') # バックエンドを指定 (GUIなし環境用)
import matplotlib.pyplot as plt
import io
import base64
from collections import Counter
import pandas as pd # 分析日時のためにインポート

class ReportGenerator:
    def generate_sentiment_pie_chart(self, sentiment_counts, total_analyzed, keyword):
        """感情分析結果の円グラフを生成し、Base64エンコードされた画像文字列を返す"""
        if total_analyzed == 0:
            return None

        labels = []
        sizes = []
        colors = []
        # explodeは一部を切り出す場合に使う (今回は使わないが参考)
        # explode_values = [] 

        positive_count = sentiment_counts.get("ポジティブ", 0)
        negative_count = sentiment_counts.get("ネガティブ", 0)
        neutral_count = sentiment_counts.get("ニュートラル", 0)

        # グラフに表示するデータを準備
        if positive_count > 0:
            labels.append(f"ポジティブ\n({positive_count/total_analyzed*100:.1f}%)")
            sizes.append(positive_count)
            colors.append('lightgreen')
            # explode_values.append(0)
        if negative_count > 0:
            labels.append(f"ネガティブ\n({negative_count/total_analyzed*100:.1f}%)")
            sizes.append(negative_count)
            colors.append('lightcoral')
            # explode_values.append(0)
        if neutral_count > 0:
            labels.append(f"ニュートラル\n({neutral_count/total_analyzed*100:.1f}%)")
            sizes.append(neutral_count)
            colors.append('lightskyblue')
            # explode_values.append(0)
        
        if not sizes: # 有効な感情分析結果がない場合
            return None

        plt.figure(figsize=(8, 7)) # グラフサイズを少し調整
        # 日本語フォント設定 (環境に合わせて調整)
        # 一般的な日本語フォントのリスト。システムにインストールされているものを使用。
        font_options = ['IPAexGothic', 'Hiragino Sans', 'Yu Gothic', 'MS Gothic', 'Noto Sans CJK JP', 'sans-serif']
        plt.rcParams['font.family'] = font_options
        
        # autopct でパーセンテージ表示、startangle で開始角度調整
        # wedgeprops で円グラフの線のスタイルを指定
        plt.pie(sizes, 
                labels=labels, 
                colors=colors, 
                autopct='%1.1f%%', 
                startangle=90, 
                pctdistance=0.85, # パーセンテージ表示位置
                labeldistance=1.05, # ラベル表示位置
                wedgeprops={'linewidth': 0.5, 'edgecolor': 'grey'}) # 境界線
        
        plt.axis('equal')  # 円を真円に
        plt.title(f"「{keyword}」に関する感情分析結果", pad=20, fontsize=14) # タイトルにキーワード追加、フォントサイズ調整

        # 画像をメモリ上のバッファに保存
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100) # dpiで解像度調整
        img_buffer.seek(0)
        
        # Base64エンコード
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close() # メモリ解放のため必ず閉じる
        
        return f"data:image/png;base64,{img_base64}"

    def generate_report(self, keyword, analyzed_tweets, positive_examples=5, negative_examples=5, neutral_examples=3):
        """分析結果のテキストレポートと、感情分析円グラフのBase64文字列を含む辞書を返す"""
        report_output = {
            "text_report": "",
            "pie_chart_base64": None
        }

        if not analyzed_tweets:
            report_output["text_report"] = "該当するツイートは見つかりませんでした。"
            return report_output
        
        sentiments = [tweet['sentiment']['label'] for tweet in analyzed_tweets if tweet.get('sentiment') and tweet['sentiment'].get('label') != 'ERROR']
        sentiment_counts = Counter(sentiments)

        total_analyzed = len(sentiments) # エラーを除いた有効な分析数
        positive_count = sentiment_counts.get("ポジティブ", 0)
        negative_count = sentiment_counts.get("ネガティブ", 0)
        neutral_count = sentiment_counts.get("ニュートラル", 0) 

        positive_share = (positive_count / total_analyzed) * 100 if total_analyzed > 0 else 0
        negative_share = (negative_count / total_analyzed) * 100 if total_analyzed > 0 else 0
        neutral_share = (neutral_count / total_analyzed) * 100 if total_analyzed > 0 else 0
         
        report_parts = []
        report_parts.append(f"調査テーマ: 「{keyword}」に関するSNSの反応")
        report_parts.append("=" * 40) # 区切り線の長さを調整
        report_parts.append(f"分析対象ツイート数 (有効な感情分析): {total_analyzed}件")
        report_parts.append(f"  - ポジティブな意見の割合: {positive_share:.2f}% ({positive_count}件)")
        report_parts.append(f"  - ネガティブな意見の割合: {negative_share:.2f}% ({negative_count}件)")
        report_parts.append(f"  - ニュートラルな意見の割合: {neutral_share:.2f}% ({neutral_count}件)")
        report_parts.append("-" * 40)

        report_parts.append("\n【ポジティブな意見の例】:")
        pos_examples_found = 0
        # スコアが高い順にソートして表示する例
        positive_tweets = sorted(
            [t for t in analyzed_tweets if t.get('sentiment') and t['sentiment']['label'] == "ポジティブ"],
            key=lambda x: x['sentiment'].get('score', 0), reverse=True
        )
        for tweet in positive_tweets:
            if pos_examples_found < positive_examples:
                report_parts.append(f"  - 「{tweet['cleaned_text']}」 (スコア: {tweet['sentiment'].get('score', 'N/A'):.2f}, 出所: {tweet['source_url']})")
                pos_examples_found += 1
        if pos_examples_found == 0:
            report_parts.append("  (該当するポジティブな意見は見つかりませんでした)")

        report_parts.append("\n【ネガティブな意見の例】:")
        neg_examples_found = 0
        negative_tweets = sorted(
            [t for t in analyzed_tweets if t.get('sentiment') and t['sentiment']['label'] == "ネガティブ"],
            key=lambda x: x['sentiment'].get('score', 0), reverse=True
        )
        for tweet in negative_tweets:
            if neg_examples_found < negative_examples:
                report_parts.append(f"  - 「{tweet['cleaned_text']}」 (スコア: {tweet['sentiment'].get('score', 'N/A'):.2f}, 出所: {tweet['source_url']})")
                neg_examples_found += 1
        if neg_examples_found == 0:
            report_parts.append("  (該当するネガティブな意見は見つかりませんでした)")
        
        report_parts.append("\n【ニュートラルな意見の例】:")
        neu_examples_found = 0
        neutral_tweets = sorted(
            [t for t in analyzed_tweets if t.get('sentiment') and t['sentiment']['label'] == "ニュートラル"],
            key=lambda x: x['sentiment'].get('score', 0), reverse=True
        )
        for tweet in neutral_tweets:
            if neu_examples_found < neutral_examples:
                report_parts.append(f"  - 「{tweet['cleaned_text']}」 (スコア: {tweet['sentiment'].get('score', 'N/A'):.2f}, 出所: {tweet['source_url']})")
                neu_examples_found += 1
        if neu_examples_found == 0:
            report_parts.append("  (該当するニュートラルな意見は見つかりませんでした)")
        
        report_parts.append("\n" + "=" * 40)
        try:
            # タイムゾーンを指定して現在時刻を取得
            analysis_time = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S %Z')
        except Exception: # pandas がない場合などのフォールバック
            import datetime
            analysis_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_parts.append(f"分析日時: {analysis_time}")

        report_output["text_report"] = "\n".join(report_parts)
        
        # 円グラフを生成して辞書に追加
        report_output["pie_chart_base64"] = self.generate_sentiment_pie_chart(sentiment_counts, total_analyzed, keyword)
        
        return report_output

# テスト用
if __name__ == '__main__':
    reporter = ReportGenerator()
    
    # 3値分類を想定したサンプルデータ
    sample_analyzed_tweets_3value = [
        {'cleaned_text': 'AIエージェントは未来を変える！期待大！素晴らしい！', 'sentiment': {'label': 'ポジティブ', 'score': 0.98}, 'source_url': 'http://example.com/pos1'},
        {'cleaned_text': 'この新機能、本当に使いやすい。ありがとう！', 'sentiment': {'label': 'ポジティブ', 'score': 0.92}, 'source_url': 'http://example.com/pos2'},
        {'cleaned_text': '雇用が奪われるのではないかと心配だ。対策が必要。', 'sentiment': {'label': 'ネガティブ', 'score': 0.85}, 'source_url': 'http://example.com/neg1'},
        {'cleaned_text': 'AIの進化は速すぎて怖い。倫理的な問題はどうなるのか。', 'sentiment': {'label': 'ネガティブ', 'score': 0.91}, 'source_url': 'http://example.com/neg2'},
        {'cleaned_text': '特に良くも悪くもない。普通だと思う。', 'sentiment': {'label': 'ニュートラル', 'score': 0.75}, 'source_url': 'http://example.com/neu1'},
        {'cleaned_text': 'AIエージェントに関するニュースを見た。', 'sentiment': {'label': 'ニュートラル', 'score': 0.60}, 'source_url': 'http://example.com/neu2'},
        {'cleaned_text': 'これについてはコメントを控えます。', 'sentiment': {'label': 'ニュートラル', 'score': 0.55}, 'source_url': 'http://example.com/neu3'},
        {'cleaned_text': '便利になるのは良いこと。期待しています。', 'sentiment': {'label': 'ポジティブ', 'score': 0.78}, 'source_url': 'http://example.com/pos3'},
        {'cleaned_text': 'バグが多い。早く修正してほしい。', 'sentiment': {'label': 'ネガティブ', 'score': 0.95}, 'source_url': 'http://example.com/neg3'},
        {'cleaned_text': 'まあまあかな。感動はなかった。', 'sentiment': {'label': 'ニュートラル', 'score': 0.68}, 'source_url': 'http://example.com/neu4'},
        {'cleaned_text': '分析エラーのテストケース', 'sentiment': {'label': 'ERROR', 'score': 0.0}, 'source_url': 'http://example.com/err1'}, # エラーケース
    ]

    keyword_to_search = "AIエージェントと雇用"
    report_data = reporter.generate_report(keyword_to_search, sample_analyzed_tweets_3value, positive_examples=3, negative_examples=3, neutral_examples=2)
    
    print("【テキストレポート】")
    print(report_data["text_report"])
    
    if report_data["pie_chart_base64"]:
        print("\n【円グラフ (Base64エンコード文字列)】")
        print(report_data["pie_chart_base64"][:100] + "...") # 長いので先頭部分のみ表示
        
        # Base64デコードしてHTMLファイルとして保存し、ブラウザで確認する例
        try:
            with open("sentiment_pie_chart.html", "w", encoding="utf-8") as f:
                f.write(f"<h1>{keyword_to_search} - 感情分析円グラフ</h1>")
                f.write(f"<img src='{report_data['pie_chart_base64']}' alt='感情分析円グラフ'>")
            print("\n円グラフを sentiment_pie_chart.html に保存しました。ブラウザで開いて確認してください。")
        except Exception as e:
            print(f"\nHTMLファイルの保存中にエラーが発生しました: {e}")
    else:
        print("\n円グラフは生成されませんでした。")