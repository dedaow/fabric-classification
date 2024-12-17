import pandas as pd


def excel_to_csv(root_path, name, column, num, data_length):
    # Read data from Excel file
    df = pd.read_excel(root_path)

    # Get data from column B starting from row 5
    data = df.iloc[4:, column]
    # Get nan index
    # nan_index = data.isna().idxmax()
    nan_index = num
    print(f'nums: {nan_index}')

    data_nums = nan_index // data_length * data_length
    # Split data into chunks of 500 rows
    chunks = [data[i:i+data_length] for i in range(0, data_nums, data_length)]

    # Save each chunk to a separate CSV file
    for i, chunk in enumerate(chunks):
        chunk.to_csv(f'csv_files/{name}_{i+1}.csv', index=False)
    print(f'Saved {name} data to CSV files.')


if __name__ == '__main__':
    root_path = 'machine learning data.xlsx'
    data_length = 300
    names = ['glass', 'white_foam', 'blue_foam', 'woven_fabric',
             'gray_sponge', 'ribbed_fabric', 'yarn_screen', 'glossy_wood', 'rough_wood',
             'twill', 'Popline', 'Polyester', 'cotton',
             'olyester', 'Dobby', 'Nylon', 'Stretch',
             'Crimp', 'plush', 'Combed', 'Tulle ']

    nums = [38602, 49224, 59549, 52639, 47194, 52807, 47806, 52200, 70477, 45103, 53759, 41079, 54424, 47220, 52786, 46944, 53062, 37950, 47730, 44888, 55118,]

    for i in range(0, len(names)):
        column = i * 3 + 1
        excel_to_csv(root_path, names[i], column, nums[i], data_length)
