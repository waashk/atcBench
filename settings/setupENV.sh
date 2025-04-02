

echo "Setting ENV"
python3.10 -m venv env

printf "\nexport WORKDIR=`dirname $PWD`" >> env/bin/activate

source env/bin/activate

pip install --upgrade pip wheel setuptools

pip install -r requirements.txt

deactivate