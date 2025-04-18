from huggingface_hub import scan_cache_dir

delete_strategy = scan_cache_dir().delete_revisions(
    "ea3cbb625c07501cd9f13d873d9679d30d622404",
)
delete_strategy.execute()
