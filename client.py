import requests
import json

if __name__ == '__main__':
    s = '山田さんが[MASK]を見たのはこれが初めてでした。巨大だった。'
    res = requests.post('http://127.0.0.1:5000/', data=json.dumps({'sent': s}))
    print(res.text)