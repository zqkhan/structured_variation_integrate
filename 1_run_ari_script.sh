source activate pytorch
for P in 2 3 6 7 9 10 11 12 13 14 15 17 18 19 21 22 23 24 25 26 27 28 29 32 33 34 36 37 38 40 43 45 46 47 49 50 51 54 55 56 58 59 60 61 62 63 64 65 66 67 68
do
	python main_ari.py --participant $P > output$P.txt &
done
echo "Running scripts in parallel"
wait
echo "Script done running"