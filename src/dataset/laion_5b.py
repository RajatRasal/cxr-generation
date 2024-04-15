"""
https://lightning.ai/lightning-ai/studios/download-stream-400m-images-text
"""
import os, math, io, concurrent, time, PIL
from litdata import optimize
from litdata.processing.readers import ParquetReader
from litdata.processing.utilities import make_request, get_worker_rank, catch


def download_image_and_prepare(row):
    image_id, url, text, _, _, image_license, nsfw, similarity = row
    data = make_request(url, timeout=1.5)
    buff = io.BytesIO()
    PIL.Image.open(data).convert('RGB').save(buff, quality=80, format='JPEG')
    buff.seek(0)
    img = buff.read()
    return [int(image_id), img, str(text), str(image_license), str(nsfw), float(similarity)]


def is_valid(row):
    try:
        return int(row[0]) and isinstance(row[2], str) and row[2] and isinstance(row[1], str) and row[1].startswith("http") and isinstance(row[5], str) and isinstance(row[6], str) and not math.isnan(float(row[7]))
    except:
        return False


# Define the class to fetch the image and serialize it back into Lightning Streaming format
class ImageFetcher:

    def __init__(self, max_threads=os.cpu_count()):
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_threads)
        # Used to track metrics
        self.stored = self.skipped = self.last_stored = 0
        self.initial_time = self.last_time = time.time()

    @property
    def total(self):
        return self.stored + self.skipped

    @property
    def success(self):
        return self.stored / self.total

    @property
    def avg_speed(self):
        return self.stored / (time.time() - self.initial_time)

    @property
    def curr_speed(self):
        return (self.stored - self.last_stored) / (time.time() - self.last_time)

    def log(self):
        if self.total % 1000 == 0:
            print(f"RANK {get_worker_rank()}: success {round(self.success * 100, 2)}, avg_speed {round(self.avg_speed, 2)} curr_speed {round(self.curr_speed, 2)} total {self.total}")
            self.last_time = time.time()
            self.last_stored = self.stored

    # THIS IS THE METHOD CALLED BY THE OPTIMIZE OPERATOR
    def __call__(self, df):
        for rows in df.iter_batches(batch_size=2048):
            rows = [row for row in rows.to_pandas().values.tolist() if is_valid(row) is True]
            futures = [self.thread_pool.submit(catch(download_image_and_prepare), row) for row in rows] 
            for future in concurrent.futures.as_completed(futures):
                try:
                    data, err = future.result()
                    self.log()

                    if data is None:
                        self.skipped += 1
                        continue

                    yield data
                    self.stored += 1

                except Exception as err:
                    self.skipped += 1


def main():
    input_dir = "/data/laion_400m/parquet"
    output_dir = "/data/laion_400m/chunks"
    parquet_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])

    # Use optimize to apply the Image Fetcher over the parquet files.
    optimize(
        fn=ImageFetcher(max_threads=16),
        inputs=parquet_files[:3],
        output_dir=output_dir,
        num_workers=os.cpu_count(),
        reader=ParquetReader("cache", num_rows=32768, to_pandas=False), # Splits the parquet files into smaller ones to ease parallelization
        chunk_bytes="64MB",
        num_downloaders=0,
        # num_nodes=32,
        # machine=Machine.DATA_PREP,
    )
