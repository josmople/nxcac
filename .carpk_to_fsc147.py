import carpk


ds = carpk.build_cached_dataset(".temp/carpk_cache")

for image, patches, target in ds:
    pass
