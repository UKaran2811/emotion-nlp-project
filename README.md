# Emotion Detection in Multilingual Social Media Text Using NLP
Darshil Maniya | 12302130601012

pip install -r requirements.txt
python common.py                                      # list language configs
python baseline.py --train_langs eng hin deu          # TF-IDF baseline + improved
python xlmr.py --train_langs eng hin deu --epochs 4   # XLM-R (use a GPU / Colab)
python report.py --tag multi                          # tables, charts, error analysis

Zero-shot: add --train_langs eng --test_langs hin --tag zs_eng2hin to baseline.py / xlmr.py / report.py.
Replace the language codes with ones printed by `python common.py`.
