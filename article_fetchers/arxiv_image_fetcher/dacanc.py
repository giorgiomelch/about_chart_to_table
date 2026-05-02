from pathlib import Path

path = Path("./images")
print(len(list(path.iterdir())))
