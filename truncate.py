from multiprocessing import Pool

import utils
import pandas as pd
import os

def truncate(trajectory, threshold = 25, num = 1, unit = 'm'):
    trans = utils.lat_long2meter()
    past = trajectory[0]
    count = 0
    tmp = [0]
    trajectories = []
    for i, coord in enumerate(trajectory[1: ]):
        j = i + 1
        distance = trans(past, coord)
        if distance > threshold:
            past = coord
            count = 0
            tmp.append(j)
        else:
            count += 1
            if count >= num:
                if len(tmp) > num:
                    trajectories.append(tmp)
                past = coord
                count = 0
                tmp = [j]
            else:
                tmp.append(j)
    if len(tmp) > num:
        trajectories.append(tmp)
    
    return trajectories

def division_1e5(x):
    return (int(x[0]) / 1e5, int(x[1]) / 1e5)

def truncate_files(files,
                    legal_coord,
                    normalize,
                    header = None,
                    columns = None,
                    delimiter = ','
                    lat_name = 'Latitude',
                    long_name = 'Longitude',
                    threshold = 25,
                    num = 1,
                    unit = 'm'):
    _data_ = []
    for file in files:
        df = pd.read_csv(file, delimiter = delimiter, header = header, low_memory = False)
        if header is None:
            df.columns = columns
        trajectory = df.loc[:, [lat_name, long_name]].values
        trajectory = [normalize(coord) for coord in trajectory if legal_coord(coord)]
        try:
            trajectories = truncate(trajectory, threshold = threshold, num = num, unit = unit)
            _data_.append((df, trajectories))
        except BaseException as exception:
            print(exception)

    return _data_

def convert(data, trajectors):
    columns = data.columns
    dicts = []
    for trajectory in trajectors:
        _dict_ = {}
        trajectory_data = data.loc[trajectory, :]
        for column in columns:
            _dict_[column] = trajectory_data[column].values.tolist()
        dicts.append(_dict_)
    
    return dicts

def raw2ouput_file(files,
                    output_file,
                    legal_coord,
                    normalize,
                    sep = '|',
                    header = None,
                    columns = None,
                    delimiter = ','
                    lat_name = 'Latitude',
                    long_name = 'Longitude',
                    threshold = 25,
                    num = 1,
                    unit = 'm'):
    _data_ = truncate_files(files,
                            legal_coord,
                            normalize,
                            header = header,
                            columns = columns,
                            delimiter = delimiter,
                            lat_name = lat_name,
                            long_name = long_name,
                            threshold = threshold,
                            num = num,
                            unit = unit)
    writer = open(output_file, 'w')
    last_position = len(columns) - 1
    for j, column in enumerate(columns):
        writer.write(column)
        if j < last_position:
            writer.write(sep)
        else:
            writer.write('\n')
    i = 0
    to_string = utils.list2string
    for data, trajectories in _data_:
        dicts = convert(data, trajectories)
        for _dict_ in dicts:
            for j, column in enumerate(columns):
                writer.write(to_string(_dict_[column]))
                if j < last_position:
                    writer.write(sep)
                else:
                    writer.write('\n')
        i += 1

def raw2trajectories(input_dir,
                    prefix,
                    postfix,
                    output_dir,
                    legal_coord,
                    normalize,
                    files_pre_worker = 50,
                    n_workers = 6,
                    sep = '|',
                    header = None,
                    columns = None,
                    delimiter = ','
                    lat_name = 'Latitude',
                    long_name = 'Longitude',
                    threshold = 25,
                    num = 1,
                    unit = 'm'):
    files = utils.get_file_list(input_dir, prefix, postfix)
    files = [os.path.join(output_dir, file) for file in files]
    pool = Pool(n_workers)
    for m in range(int((len(files) / files_pre_worker))):
        worker_files = [m * files_pre_worker: (m + 1) * files_pre_worker]
        pool.apply_async(raw2ouput_file, (worker_files,
                                        os.path.join(output_dir, str(m) + '.csv'),
                                        legal_coord,
                                        normalize,
                                        sep,
                                        header,
                                        columns,
                                        delimiter,
                                        lat_name,
                                        long_name,
                                        threshold,
                                        num,
                                        unit,))
    pool.close()
    pool.join()

if __name__ == '__main__':
    pass