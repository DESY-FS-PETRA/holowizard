from pathlib import Path
import io
import pickle
import time


class ParamsSerializer:
    def __init__(self):
        pass

    @staticmethod
    def serialize(params, path=None):
        f = io.BytesIO()
        pickle.dump(params, f)
        serialized = f.getvalue()
        if path is not None:
            with open(path, "wb") as file:
                file.write(serialized)
        return serialized

    @staticmethod
    def deserialize(data, timeout=100):
        if type(data) == str:
            path = Path(data)
            tries = 0
            while not path.exists() and tries < timeout:
                time.sleep(0.1)
                tries += 1
            if tries == timeout:
                raise RuntimeError(
                    "Reached timeout while trying to deserialize " + data
                )
            with open(data, "rb") as f:
                params = pickle.load(f)
        else:
            f = io.BytesIO(data)
            params = pickle.load(f)

        return params
