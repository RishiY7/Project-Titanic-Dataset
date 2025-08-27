def colnames(data1):
    data1.columns = data1.columns.str.lower().str.replace(" ", "_")
    return data1
