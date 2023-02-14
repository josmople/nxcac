import carpk
from tqdm import tqdm

ds = carpk.build_cached_dataloader(".temp/carpk_cache", batch_size=1)

for i in tqdm(ds):
    pass
