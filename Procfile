# This tells Cloud Run from its GitHub repo build trigger how to serve the Flask app
# gevent provides fully asynchronous unbuffered WSGI service with only 1 worker required
web: python -u -m gunicorn -b :8080 -k gevent -w 1 main:app
