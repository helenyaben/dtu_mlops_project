import wandb

wandb.init(project="test_on_mnist", entity="dtu_mlops_2023", config={
"learning_rate": 0.02,
"epochs": 30,
"batch_size": 64
})