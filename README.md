# WSG_games


Install dictionary_learning (run at the top level:)
```bash
git clone https://github.com/jkminder/dictionary_learning.git   
```

Change in line 38 of https://github.com/jkminder/dictionary_learning/blob/b4f272f7ea9917c9358c573c95c24469c320e773/dictionary_learning/utils.py#L38
```python
# np.bool: th.bool,
np.bool_: th.bool,
```
And in line 85 of https://github.com/jkminder/dictionary_learning/blob/b4f272f7ea9917c9358c573c95c24469c320e773/dictionary_learning/cache.py#L85
```python
# self._tokens = th.load(
#     os.path.join(store_dir, "tokens.pt"), weights_only=True
# ).cpu()
parent_dir = os.path.dirname(self.store_dir)  # NEW
token_file_path = os.path.join(parent_dir, "tokens.pt")  # NEW
self._tokens = th.load(
    os.path.join(token_file_path), weights_only=True
).cpu()
```
Then in /dictionary_learning
```python
uv pip install -e .
```