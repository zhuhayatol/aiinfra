from tiny_lm.data.dataloader import DataLoaderLite

def test_dataload():
    dataload = DataLoaderLite(B=2, T=4, file_name="../data/input.txt")
    x, y  = dataload.next_batch()
    print(x, y)
    assert x.shape == (2, 4)
    assert y.shape == (2, 4)