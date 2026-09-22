from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


SHUFFLE_SEED = 11
HOLDOUT_SEED = 22
VALID_TEST_SEED = 33


def login_if_token(hf_token: str | None) -> None:
    if hf_token:
        import huggingface_hub

        huggingface_hub.login(token=hf_token)


def split_train_valid_test(data: Dataset) -> DatasetDict:
    split = data.train_test_split(test_size=20000, seed=HOLDOUT_SEED)
    train = split["train"]
    valid = split["test"].train_test_split(test_size=10000, seed=VALID_TEST_SEED)
    return DatasetDict({
        "train": train,
        "valid": valid["train"],
        "test": valid["test"],
    })


def build_uniref50_splits() -> DatasetDict:
    data = load_dataset("agemagician/uniref50_09012025")
    data = data.remove_columns("id").remove_columns("name").shuffle(seed=SHUFFLE_SEED)
    data = data.rename_column("text", "sequence")
    data = concatenate_datasets([data["train"], data["validation"], data["test"]])
    return split_train_valid_test(data)


def build_omg_prot50_splits() -> DatasetDict:
    data = load_dataset("tattabio/OMG_prot50", split="train")
    data = data.remove_columns("id").shuffle(seed=SHUFFLE_SEED)
    return split_train_valid_test(data)


def build_og_prot90_splits() -> DatasetDict:
    data = load_dataset("tattabio/OG_prot90", split="train")
    data = data.remove_columns("id").shuffle(seed=SHUFFLE_SEED)
    return split_train_valid_test(data)


def push_splits(dataset: DatasetDict, repo_id: str) -> None:
    print(dataset)
    dataset.push_to_hub(repo_id)
