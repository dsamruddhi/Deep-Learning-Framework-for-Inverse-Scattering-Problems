class Config:
    config = {
        "data": {
            "input_paths": "",
            "output_paths": "",
            "standard_path": "",
            "test_size": 0.1,
            "num_samples": 5000
        },
        "model": {
            "model_path": "",
            "experiment_name": "test"
        },
        "train": {
            "initial_learning_rate": 0.0003,
            "decay_steps": 100000,
            "decay_rate": 0.8,
            "optimizer": None,
            "loss": "mean_absolute_error",
            "metrics": ["accuracy"],
            'validation_split': 0.2,
            "epochs": 100,
            "train_batch_size": 32,
            "val_batch_size": 8
        },
        "test": {
            "chi_real_path": "",
            "chi_imag_path": "",
            "output_path": None
        },
        "plot": {
            "doi_length": 1.5,
            "cmap": "jet"
        }
}