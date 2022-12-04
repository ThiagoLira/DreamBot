export FLASK_APP=app
gunicorn -b 192.168.15.2:5000 --timeout=900 sd_server:app
