import pandas as pd
import numpy as np
from src.layer0.profiler import data_profile

data = {
    "col_num": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    "col_obj": ["A", "B", "A", "B", "C", "A", "B", "C", "A", "B", "C"],
    "col_suspect": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]
}
df = pd.DataFrame(data)
profile = data_profile(df)
print(profile.to_string())
