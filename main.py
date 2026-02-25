from os import environ
if not (DEEPGRAM_API_KEY := environ.get("DEEPGRAM_API_KEY")):
    raise RuntimeError("DEEPGRAM_API_KEY is not configured; please add it to the environment")

from app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
