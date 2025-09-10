from utils.preferences_train import PreferencesTrainer

runner = PreferencesTrainer(
    log_policy_dir_list=["./results/DSAC2"],
    trained_policy_iteration_list=["2000"],
    is_init_info=False,
    save_render=False,
    legend_list=["DSAC"],
    manual = False, 
    stage = 2,
    dt=0.01, # time interval between steps
)

runner.run()
