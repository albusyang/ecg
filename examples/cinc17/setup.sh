mkdir -p data && cd data

BASE_URL="https://www.physionet.org/files/challenge-2017/1.0.0/"

wget "${BASE_URL}training2017.zip?download" -O training2017.zip
unzip training2017.zip

wget "${BASE_URL}sample2017.zip?download" -O sample2017.zip
unzip sample2017.zip

wget "${BASE_URL}REFERENCE-v3.csv" -O REFERENCE-v3.csv

cd ..

python build_datasets.py
