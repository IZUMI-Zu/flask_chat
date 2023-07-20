"""
This module serves as the entry point of a Flask application that uses Gunicorn as the WSGI HTTP 
Server. During execution, this script initializes the necessary environment settings and command 
line arguments for Gunicorn and launches the server with the specified Flask application.

This module requires the presence of 'app:app' which is the Flask app object from the module 
named "app.py". This should be customized based on the actual Flask app location in your project.

Once the module is executed, it sets the CUDA_VISIBLE_DEVICES to "0" which implies that only 
the first GPU (if available) is used by the application.

The server binds on port 5151 and would be accessible over the network (0.0.0.0 means listen 
on all available network interfaces).

Furthermore, this script assumes a CUDA environment by setting CUDA_VISIBLE_DEVICES. If the
machine does not possess a GPU or the CUDA toolkit, this environment setting may not be effective.

This script is typically used in production environments where a robust, extensible, and 
performant server like Gunicorn would be required over Flask's built-in server.
"""
import os
from waitress import serve
import app

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    serve(app.app, host='0.0.0.0', port=5151)