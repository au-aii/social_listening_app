# app/data_collector.py
import requests
import json
from config import TWITTER_BEARER_TOKEN # config.py から読み込み

class DataCollector:
    def __init__(self):
        self.bearer_token = TWITTER_BEARER_TOKEN
        self.search_url = "https://api.twitter.com/2/tweets/search/recent"

    def _bearer_oauth(self, r):
        r.headers["Authorization"] = f"Bearer {self.bearer_token}"
        r.headers["User-Agent"] = "v2RecentSearchPython"
        return r

    def search_tweets(self, keyword, max_results=100):
        params = {
            'query': f'{keyword} lang:ja -is:retweet', # 日本語、リツイート除外
            'max_results': max_results,
            'tweet.fields': 'created_at,text,author_id,id,public_metrics,source' # 取得したい情報
        }
        try:
            response = requests.get(self.search_url, auth=self._bearer_oauth, params=params)
            response.raise_for_status() # エラーがあれば例外を発生させる
            json_response = response.json()
            
            tweets = []
            if 'data' in json_response:
                for tweet_data in json_response['data']:
                    tweets.append({
                        'id': tweet_data['id'],
                        'text': tweet_data['text'],
                        'created_at': tweet_data['created_at'],
                        'author_id': tweet_data['author_id'],
                        'source_url': f"https://twitter.com/{tweet_data['author_id']}/status/{tweet_data['id']}"
                        # 必要に応じて他の情報も追加
                    })
            return tweets
        except requests.exceptions.RequestException as e:
            print(f"Error during Twitter API request: {e}")
            print(f"Response content: {response.content if 'response' in locals() else 'N/A'}") # エラー時のレスポンス内容を確認
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            print(f"Response content: {response.content if 'response' in locals() else 'N/A'}")
            return []

# テスト用
if __name__ == '__main__':
    collector = DataCollector()
    keyword_to_search = "AIエージェント 雇用"
    tweets_data = collector.search_tweets(keyword_to_search, max_results=10)
    if tweets_data:
        for tweet in tweets_data:
            print(f"ID: {tweet['id']}, Text: {tweet['text'][:50]}..., URL: {tweet['source_url']}")
    else:
        print("No tweets found or error occurred.")