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
import sys
import os
from gunicorn.app.wsgiapp import run

if __name__ == '__main__':
    """
    This is the entry point of the application when it's being run standalone rather than being 
    imported as a module.

    It sets CUDA_VISIBLE_DEVICES to "0". This ensures that only the first GPU (should the system 
    have GPUs) is visible and usable by the application.

    After that, it sets sys.argv to the gunicorn WSGI server startup command: ["gunicorn", "--bind",
    "0.0.0.0:5151", "app:app"], which means running gunicorn WSGI server, binding it to port 5151 
    accessible from any host, and using the Flask app object from the "app" module as the WSGI application.

    It then uses the `run` function of the `gunicorn.app.wsgiapp` package to execute the startup 
    command and start the application. The program exits with the exit code returned from the `run` 
    function.

    Note: 
    - This script assumes that the machine it runs on has a GPU and the CUDA toolkit installed, otherwise the 
      CUDA_VISIBLE_DEVICES may not have any effect.
    - The `app` in "app:app" refers to a Flask app instance defined in a module named "app.py". Modify this 
      accordingly based on your actual Flask app instance path.

    """
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    sys.argv = "gunicorn --bind 0.0.0.0:5151 app:app".split()
    sys.exit(run())
