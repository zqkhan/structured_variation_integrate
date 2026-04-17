# Structured variation in daily life experience within and across individuals
There are two ways to run this code:
1. On Linux, To run all (or multiple participants) in parallel, go to your command line and use "bash 1_run_ari_script.sh". Replace "pytorch" in source activate pytorch with your python environment name.
2. On Windows, To run all (or multiple participants) in parallel, go to your anaconda command prompt, activate your environment and run following command:

for %P in (2 3 6 7 9 10 11 12 13 14 15 17 18 19 21 22 23 24 25 26 27 28 29 32 33 34 36 37 38 40 43 45 46 47 49 50 51 54 55 56 58 59 60 61 62 63 64 65 66 67 68) do python main_ari.py --participant %P --use_cols All > output%P.txt
3. To run one participant at a time use python main_ari.py --participant s_id --use_cols All (e.g. for participant 2 use python main_ari.py --participant 2). 

4. for each participant two files will be generated in the 'results\' folder. one named _mdl_clusters.pk that saves the best clustering model for the K that gives min coding cost, second named mdl_costs_k_s_id.PNG which is the plot.

5. you can then run main_ari_visualizations.py to generate the summary plots. the script will automatically do this for all participants with models saved in 'results/' and output the visualizations in 'visualizations\' folder.
6. You can also run 3_main_combine_data_results.py to combine the cluster ids and the input data, the output will go to 
the results/ directory in the form PID_data_results.csv and the first column is cluster assignments. By default this will have the standardized data that is used for clustering. Set Standardize to False to create CSVs with original data.

7. To run feature importance:

Install https://github.com/zqkhan/mifs_forked/

8. Run 4_main_ari_feature_importance_fewer.py , be sure to pass the participant number to p_list and the appropriate results and visualizations directory.


Notes:
- You can use following options for --use_cols:

All : Runs all features (default)
Affect: Runs using only Valence and Arousal
Physio: Runs using only Physio
Continuous: Uses Physio + Affect
Categorical: Uses just Social, Posture, Activity

If you want to try any other combination you'd have to pass it in the form of a list e.g. for just Social and mean_SV you will pass ['Social.2', 'mean_SV'] and you'll have to pass a directory name to results_dir argument.

- You can run the python files directly too, for main_ari.py scroll down and make sure to put in
the correct participant number in the default. For example:

parser.add_argument('--participant', type=int, default=2)
parser.add_argument('--use_cols', type=int, default=['mean_SV', 'mean_IBI'])
parser.add_argument('--results_dir', type=str, default=['results_SVIBI\'])

means participant '2' will run with just mean_SV and mean_IBI and results will get deposited in 'results_SVIBI\' directory
