import os
from data.generate_sample_data import generate
from training.preprocess import preprocess


def test_generate_and_preprocess(tmp_path):
    # Generate sample data and ensure preprocess runs using UCI-like file format
    path = os.path.join(str(tmp_path), 'household_power_consumption.txt')
    from shutil import copyfile
    src = generate(path)
    assert src == path
    # Run preprocess
    out = preprocess(raw_path=src, out_path=os.path.join(str(tmp_path), 'processed.csv'))
    assert os.path.exists(out)
    # Should not contain nulls in critical columns
    import pandas as pd
    df = pd.read_csv(out)
    assert 'demand' in df.columns
    # Ensure no nulls in demand column were left after preprocessing
    assert df['demand'].isnull().sum() == 0
