#!/bin/bash

time_intervals='date_week'
echo 'CREATE TEMPLATE DATASETS'
for interval in $time_intervals
do
    suffix='monthly-medium'
    samples=500
    psteps=1
    if [ "$interval" = "date_week" ]
    then
    suffix='weekly-medium'
    psteps=20
    samples=500
    msteps=57
    fi
    echo "$interval", "$samples"
    echo '=============== Politics ==============='
    python scripts/preprocessing/supervised_dynamic_language_dataset_preprocessor.py -i data/raw/reddit/politics/submissions/aggregated/data_0_comments.csv -o data/preprocessed/reddit-50000/politics/submissions/language/num-comments-$suffix -e embeddings/glove.6B.300d.txt -min-df-tp 5  -tt-ratio 0.8 0.1 -max-doc-len 150 --reward-column num_comments -br 2 -psteps $psteps -c date_day int  --date-field-name $interval --num-workers 14 -bow-vocab-size 5000 --samples-per-timestep $samples --max-reward 50000
    echo '=============== Donald ==============='
    python scripts/preprocessing/supervised_dynamic_language_dataset_preprocessor.py -i data/raw/reddit/the_donald/submissions/aggregated/data_0_comments.csv -o data/preprocessed/reddit-50000/the_donald/submissions/language/num-comments-$suffix -e embeddings/glove.6B.300d.txt -min-df-tp 5  -tt-ratio 0.8 0.1 -max-doc-len 200 --reward-column num_comments -br 2 -psteps $psteps -c date_hour int --date-field-name $interval --num-workers 14 -bow-vocab-size 5000 --samples-per-timestep $samples --max-reward 50000
    echo '=============== Wallstreetbets ==============='
    python scripts/preprocessing/supervised_dynamic_language_dataset_preprocessor.py -i data/raw/reddit/wallstreetbets/submissions/aggregated/data_0_comments.csv -o data/preprocessed/reddit-50000/wallstreetbets/submissions/language/num-comments-$suffix -e embeddings/glove.6B.300d.txt -min-df-tp 5  -tt-ratio 0.8 0.1 -max-doc-len 500 --reward-column num_comments -br 2  -psteps $psteps -c date_month int --date-field-name $interval --num-workers 14 -bow-vocab-size 5000 --samples-per-timestep $samples --max-reward 50000
done

samples=1000
interval='date_month'
suffix='monthly-medium'
psteps=20
echo '=============== Science ==============='
python scripts/preprocessing/supervised_dynamic_language_dataset_preprocessor.py -i data/raw/reddit/askscience/submissions/aggregated/data_0_comments.csv -o data/preprocessed/reddit-50000/askscience/submissions/language/num-comments-$suffix -e embeddings/glove.6B.300d.txt -min-df-tp 5  -tt-ratio 0.8 0.1 -max-doc-len 120 --reward-column num_comments -br 2 -psteps $psteps -c date_month int --date-field-name $interval --num-workers 14  -bow-vocab-size 5000 --samples-per-timestep $samples --max-reward 50000
echo ''
