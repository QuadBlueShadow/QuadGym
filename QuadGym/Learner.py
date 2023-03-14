from InstanceManager import LearnerInstance

#The check timer is the amount of minutes before we check for data
learner = LearnerInstance(check_timer=10, lr=0.005, n_epochs=1)

data_save_dir = f"C:/example_data/"
model_save_dir = f"C:/example_models/main_save.zip"

learner.run(model_save_dir, data_save_dir)