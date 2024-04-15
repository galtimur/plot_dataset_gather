2. generate_tasks_by_GPT.py - generate tasks for drawing the plots. Input - plot.py, plot(s).png, data_descr.txt. Input is taken from manually validated dataset (dataset_valid_step_1)
output results to the jsonl file, storing raw responses
3. process_step_3_gather_dps.py - processes GPT responses and generates final dataset that contains
 ["plot.py", "data_descr.txt", "data.csv", "plot_original.py", "task.json", "plot(s).png"]