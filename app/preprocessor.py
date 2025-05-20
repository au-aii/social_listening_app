# app/preprocessor.py
import re
from janome.tokenizer import Tokenizer # 例としてJanomeを使用

class Preprocessor:
    def __init__(self):
        self.tokenizer = Tokenizer() # 形態素解析器のインスタンス化

    def clean_text(self, text):
        # URLの除去
        text = re.sub(r'https?://\S+', '', text)
        # ハッシュタグの除去 (タグ自体は残しても良いが、ここでは#記号とタグ名を消す)
        text = re.sub(r'#\S+', '', text)
        # メンションの除去
        text = re.sub(r'@\S+', '', text)
        # HTMLタグの除去 (スクレイピングした場合など)
        text = re.sub(r'<[^>]+>', '', text)
        # 絵文字の除去 (必要に応じて)
        # ... (絵文字除去ライブラリや正規表現を使用)
        # 空白の正規化
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def anonymize_user_info(self, tweet_data):
        # 実際にはユーザーIDなどもマスキング対象だが、ここでは簡易的に
        # 収集データ構造に合わせて調整
        if 'author_id' in tweet_data:
            tweet_data['author_id'] = "ANONYMIZED_USER"
        return tweet_data

    def tokenize(self, text):
        # 形態素解析して単語リスト（または特定の品詞のみ）を返す
        tokens = [token.surface for token in self.tokenizer.tokenize(text)]
        return tokens

    def preprocess_tweet(self, tweet_data):
        processed_tweet = tweet_data.copy()
        processed_tweet['cleaned_text'] = self.clean_text(tweet_data['text'])
        # processed_tweet['tokens'] = self.tokenize(processed_tweet['cleaned_text']) # 必要なら
        processed_tweet = self.anonymize_user_info(processed_tweet)
        return processed_tweet

# テスト用
if __name__ == '__main__':
    preproc = Preprocessor()
    sample_tweet = {
        'id': '123',
        'text': 'これはテストツイートです！AIエージェントすごい #AI https://example.com @user123',
        'created_at': '2023-01-01T00:00:00Z',
        'author_id': 'user_abc'
    }
    processed = preproc.preprocess_tweet(sample_tweet)
    print(f"Original: {sample_tweet['text']}")
    print(f"Cleaned: {processed['cleaned_text']}")
    print(f"Anonymized Author: {processed['author_id']}")