# Run 5 tests with random data
for i in {1..5}
do
    echo "Running test $i"
    python per_seq_cp_fwd_correct_check.py --fix_seed=0
done