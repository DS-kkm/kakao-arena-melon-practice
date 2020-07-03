#사용자함수:check_dtypes()
#각 칼럼 내 데이터의 최소, 최대 범위를 계산해 적절한 data type을 찾아내는 함수
#각 컬럼의 데이터 형식을 모를 때 자동으로 체크하여 사용하고 데이터를 불러올 때 체크된 데이터 형식으로 데이터를 불러옵니다
#형식을 지정하지 않으면 가장 메모리를 많이 차지하는 방식으로 데이터를 불러오기 때문에
#형식을 지정해서 데이터를 불러오면 메모리를 줄여 큰 데이터도 불러올 수 있습니다


def check_dtypes(file_path):
    print(file_path)
    tmp = pd.read_csv(file_path, nrows=0)
    col_dtypes = {}
    for col in tmp.columns:
        df = pd.read_csv(file_path, usecols=[col])
        dtype = df[col].dtype

        if dtype == 'int' or dtype == 'float':
            c_min = df[col].min()
            c_max = df[col].max()
        elif dtype == 'object':
            n_unique = df[col].unique()
            threshold = n_unique / df.shape[0]

        if dtype == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                col_dtype = 'int8'
            elif c_min > np.iinfo(np.unit8).min and c_max < np.iinfo(np.unit8).max:
                col_dtype = 'unit8'
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                col_dtype = 'int16'
            elif c_min > np.iinfo(np.unit16).min and c_max < np.iinfo(np.unit16).max:
                col_dtype = 'unit16'
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                col_dtype = 'int32'
            elif c_min > np.iinfo(np.unit32).min and c_max < np.iinfo(np.unit32).max:
                col_dtype = 'unit32'
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                col_dtype = 'int64'
            elif c_min > np.iinfo(np.unit64).min and c_max < np.iinfo(np.unit64).max:
                col_dtype = 'unit64'

        elif dtype == 'float':
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                col_dtype = 'float32'
            else:
                dol_dtype = 'float64'

        elif dtype == 'object':
            if threshold > 0.7:
                col_dtype = 'object'
            else:
                col_dtype = 'category'

        col_dtypes[col] = col_dtype

    return col_dtypes
