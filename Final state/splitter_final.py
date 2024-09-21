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


def split(col_names, filename, output_file, reduction_factor, sarcastic_proportion, ram_limit_gb, chunk_size=None):
    """
    returns dataset shortened by reduction_factor containing default 40% sarcastic replies
    """
    if chunk_size is None:
        chunk_size = estimate_chunk_size(filename, ram_limit_gb)
    print("chunk size: ", chunk_size)

    # Count total rows to compute the target sample size
    total_rows = get_total_rows(filename, chunk_size)
    target_rows = total_rows // reduction_factor

    # Determine number of sarcastics and genuine rows required
    sarcastics_needed = int(target_rows * sarcastic_proportion)
    genuine_needed = target_rows - sarcastics_needed

    # Collect row indices for sampling
    indices = np.arange(total_rows)
    np.random.shuffle(indices)  # Shuffle indices for randomness

    # Read the dataset in chunks and sample the rows randomly
    sarcastics = []
    genuine_samples = []

    with pd.read_csv(filename, sep="\t", chunksize=chunk_size, on_bad_lines='skip', names=col_names) as reader:
        for chunk in reader:
            chunk_indices = chunk.index.values
            for idx in chunk_indices:
                if len(sarcastics) >= sarcastics_needed and len(genuine_samples) >= genuine_needed:
                    break

                if chunk.loc[idx, "label"] == 1 and len(sarcastics) < sarcastics_needed:
                    sarcastics.append(chunk.loc[idx])
                elif chunk.loc[idx, "label"] == 0 and len(genuine_samples) < genuine_needed:
                    genuine_samples.append(chunk.loc[idx])

            if len(sarcastics) >= sarcastics_needed and len(genuine_samples) >= genuine_needed:
                break

    if not sarcastics:
        raise ValueError("No sarcastic entries labelled as 1 found in the dataset")

    if not genuine_samples:
        raise ValueError("No genuine entries labelled as 0 found in the dataset")

    sarcastics_df = pd.DataFrame(sarcastics, columns=col_names)
    genuine_df = pd.DataFrame(genuine_samples, columns=col_names)

    result_chunk = pd.concat([sarcastics_df, genuine_df])

    # Shuffling result to ensure mixed order
    result_chunk = result_chunk.sample(frac=1, random_state=42).reset_index(drop=True)

    # Writing result
    result_chunk.to_csv(output_file, sep="|", mode='w', index=False, header=True)

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
    
