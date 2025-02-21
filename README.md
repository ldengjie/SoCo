conda create --name soco python=3.10

conda activate soco

pip install torch==2.5.1 transformers==4.43.2 lm_eval==0.4.7 accelerate==1.2.1 sentencepiece==0.2.0 protobuf==3.19.6

python infer.py

python eval.py
