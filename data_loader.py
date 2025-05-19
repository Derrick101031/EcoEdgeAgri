# data_loader.py
import requests, pandas as pd, io, datetime as dt

def fetch_thingspeak(channel_id:int, api_key:str)->pd.DataFrame:
    # 1 day window
    now = dt.datetime.utcnow()
    start = (now - dt.timedelta(days=1)).strftime('%Y-%m-%d%%20%H:%M:%S')
    url = (f"https://api.thingspeak.com/channels/{channel_id}/feeds.csv?"
           f"api_key={api_key}&start={start}")
    csv = requests.get(url, timeout=30).text
    return pd.read_csv(io.StringIO(csv))
