from utils.sys_run import PolicyRunner

runner = PolicyRunner(
    log_policy_dir_list=["./results/DSAC2"],
    trained_policy_iteration_list=["3000"],
    is_init_info=False,
    save_render=False,
    legend_list=["DSAC"],
    manual = True, 
    dt=0.01, # time interval between steps
)

runner.run()
