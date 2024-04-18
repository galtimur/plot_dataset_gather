import pandas as pd
def get_pycharm_dataframe_description(task_text: str, df: pd.DataFrame) -> str:

    # task_text is dummy

    descr_lines = [f'Number of rows in DataFrame: {len(df)}']
    descr_lines.append('DataFrame has the following columns:')
    for col in df.columns:
        types_set = set(df.loc[df[col].notna(), col].apply(type))
        types_list = [str(type_.__name__) for type_ in types_set]
        if len(types_list) == 1:
            col_types = types_list.pop()
        else:
            col_types = str(set(types_list)).replace('"', '').replace('\'', '')
        descr = f'{col} of type {col_types}. Count: {df[col].count()}'
        if str(df[col].dtype).startswith(('int', 'float')):
            mean = f'{df[col].mean():.6}'
            std = f'{df[col].std():.6}'
            if str(df[col].dtype).startswith('int'):
                minimum = f'{df[col].min()}'
                maximum = f'{df[col].max()}'
            else:
                minimum = f'{df[col].min():.6}'
                maximum = f'{df[col].max():.6}'
            descr = descr + f', Mean: {mean}, Std. Deviation: {std}, Min: {minimum}, Max: {maximum}'
        descr_lines.append(descr)
    return '\n'.join(descr_lines)
