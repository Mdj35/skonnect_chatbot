from app import app

# Expose the Flask app as 'app' for gunicorn/uWSGI
application = app