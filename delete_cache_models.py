from huggingface_hub import scan_cache_dir

delete_strategy = scan_cache_dir().delete_revisions(
    "86b5e0934494bd15c9632b12f734a8a67f723594",
    "3f2e93603aaa5dd142f27d34b06dfa2b6e97b8be",
)
delete_strategy.execute()
