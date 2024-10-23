mode = "cluster"
params = {
    "cluster": {
        "EEV_Samples": 10000,
        "WS_Samples": 10000,
        "TS_Naive_Samples": 100,
        "TS_Naive_Counts": 100,
        "TS_LShaped_Samples": 1000,
        "TS_LShaped_Iterations": 100,
        "TS_LShaped_Counts": 10,
    },
    "local": {
        "EEV_Samples": 100,
        "WS_Samples": 100,
        "TS_Naive_Samples": 50,
        "TS_Naive_Counts": 10,
        "TS_LShaped_Samples": 100,
        "TS_LShaped_Iterations": 50,
        "TS_LShaped_Counts": 10,
    },
    "test": {
        "EEV_Samples": 2,
        "WS_Samples": 2,
        "TS_Naive_Samples": 2,
        "TS_Naive_Counts": 2,
        "TS_LShaped_Samples": 2,
        "TS_LShaped_Iterations": 2,
        "TS_LShaped_Counts": 2,
    }
}.get(mode)