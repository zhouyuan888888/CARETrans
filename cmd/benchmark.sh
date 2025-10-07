python3 benchmark.py \
--bench inference \
--model CARETrans_S0  \
--img-size 224 -b 64 --no-retry

python3 benchmark.py \
--bench inference \
--model CARETrans_S1  \
--img-size 224 -b 64 --no-retry

python3 benchmark.py \
--bench inference \
--model CARETrans_S2  \
--img-size 224 -b 64 --no-retry