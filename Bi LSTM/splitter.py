import pandas as pd
import numpy as np

LENTOT = 533246031
LENSARC = 1341171


def estimate_chunk_size(filename, ram_limit_gb, memory_margin=0.4):
    """
    Estimating reasonable chunk size for processing of the massive archive
    """
    sample_size = 1000
    sample = pd.read_csv(filename,sep = "\t", nrows = sample_size, on_bad_lines = 'skip')
    sample_memory_usage = sample.memory_usage(deep = True).sum()
    memory_per_row = sample_memory_usage / sample_size

    available_memory = ram_limit_gb*(1 - memory_margin) * (1024**3)
    estimated_chunk_size = int(available_memory//memory_per_row)

    return max(1, estimated_chunk_size)

def get_total_rows(filename, chunk_size):
    """
    return total rows of dataset
    """
    total_rows = 0
    for chunk in pd.read_csv(filename,sep = "\t", chunksize = chunk_size, on_bad_lines = 'skip'):
        total_rows += len(chunk)
    return total_rows


def split(col_names, filename, output_file, reduction_factor, sarcastic_proportion, ram_limit_gb, chunk_size = None):
    """
    returns dataset shortened by reduction_factor containing default 40% sarcastic replies
    """

    if chunk_size is None:
        chunk_size = estimate_chunk_size(filename, ram_limit_gb)
    print("chunk size: ", chunk_size)

    total_rows = get_total_rows(filename, chunk_size)
    target_rows = total_rows // reduction_factor

    sarcastics = []

    with open(output_file, mode = 'w') as writer:
        chunk_number = 0

        for chunk in pd.read_csv(filename,sep = "\t", chunksize = chunk_size, on_bad_lines = 'skip', names = col_names):
            print("Sarcastics, chunk number: ", chunk_number)
            chunk_number += 1
            sarcastics_chunk = chunk[chunk["label"] == 1]
            sarcastics.append(sarcastics_chunk)
            if sum(len(df) for df in sarcastics) >= target_rows:
                break

        if not sarcastics:
            raise ValueError("No sarcastic entries labelled as 1 found in the dataset")

        sarcastics = pd.concat(sarcastics)
        num_sarc = len(sarcastics)
        num_gen_to_sample = int(num_sarc * (1 - sarcastic_proportion)/sarcastic_proportion)

        sampled_genuine = []
        chunk_number = 0
        for chunk in pd.read_csv(filename,sep = "\t", chunksize = chunk_size, on_bad_lines = 'skip', names = col_names):
            print("Genuines, chunk number: ", chunk_number)
            chunk_number += 1
            genuine_chunk = chunk[chunk["label"] == 0]
            if len(genuine_chunk) > 0:
                sampled_genuine_chunk = genuine_chunk.sample(n = min(num_gen_to_sample, len(genuine_chunk)), random_state = 42)
                sampled_genuine.append(sampled_genuine_chunk)
                num_gen_to_sample -= len(sampled_genuine_chunk)
                if num_gen_to_sample <= 0:
                    break
        print(type(sampled_genuine))
        print(len(sampled_genuine))
        
        if not sampled_genuine:
            raise ValueError("No genuine entries labelled as 0 found in the dataset")

        sampled_genuine = pd.concat(sampled_genuine)

        result_chunk = pd.concat([sarcastics, sampled_genuine])

        # Shuffling result
        result_chunk = result_chunk.sample(frac = 1, random_state = 42).reset_index(drop = True)

        # Writing result
        result_chunk.to_csv(output_file, sep = "|", mode = 'w', index = False, header = True)
        



def main():
    filename = str(input("Filename: "))
    output_file = str(input("Output filename (without extension): ")) + ".csv"
    reduction = int(input("Reduction factor: "))
    proportion = float(input("Sarcastic proportion in decimal value: "))
    ram_lim = int(input("Ram overload limitation in gb: "))

    col_names = ["label", "comment", "author", "subreddit", "score", "ups", "downs", "date created", "utc", "parent comment", "id", "link_id"]

    split(col_names, filename, output_file, reduction, proportion, ram_lim)

if __name__ == "__main__":
    main()
    
