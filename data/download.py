from datasets import load_dataset


dataset = load_dataset("araoye01/sem_eval_2018_task_1", trust_remote_code=True)

print(dataset['train'].head())

