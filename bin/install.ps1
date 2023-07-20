$VENV_NAME="flask_venv"
$MODEL_FILE_NAME="model.tar.gz"
$TAG="v0.2.0"

py -m venv "${VENV_NAME}"
. "${VENV_NAME}/Scripts/activate.ps1"
py -m pip install --upgrade pip
py -m pip install --no-cache-dir -r requirements.txt

Invoke-WebRequest -Uri "https://github.com/IZUMI-Zu/flask_chat/releases/download/${TAG}/${MODEL_FILE_NAME}" -OutFile "${MODEL_FILE_NAME}"
tar -zxvf "${MODEL_FILE_NAME}"
Remove-Item -path "${MODEL_FILE_NAME}"

Write-Output "
To activate the virtual environment, run:

    $env:VENV_NAME\\Scripts\\activate.ps1

To deactivate the virtual environment, run:

    deactivate
"