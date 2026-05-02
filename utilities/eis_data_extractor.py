import numpy as np


def extract_eis_capacity_dataset(data, cell_id, soc_filter=None):
    if cell_id not in data:
        raise ValueError(f"Cell ID '{cell_id}' not found.")

    X1_list = []
    X2_list = []
    Y_list = []
    cycles = []
    socs = []

    for nCycle in sorted(data[cell_id]):
        cycle_data = data[cell_id][nCycle]
        valid_socs = sorted(cycle_data.keys())
        if soc_filter is not None:
            valid_socs = [s for s in valid_socs if s == soc_filter]

        for soc in valid_socs:
            item = cycle_data[soc]
            eis = item['eis'][:, -2:]
            eis_image = item['eis_image']
            soh = item['ca'][0, 1]

            X1_list.append(eis)
            X2_list.append(eis_image)
            Y_list.append(soh)
            cycles.append(nCycle)
            socs.append(soc)

    if not X1_list:
        print(f"No valid EIS/CA data found for cell {cell_id}, continue!")

    # X = np.stack(X_list)  # shape: (n_samples, n_freq, 2)
    # Y = np.array(Y_list)

    return X1_list, X2_list, Y_list, {'cycles': cycles, 'socs': socs}


class DataExtractorSpecific:
    def __init__(self, dict_data, soc_filter=None):
        self.data = dict_data
        self.soc_filter = soc_filter
        self.cell_ids = self.get_all_cell_ids()
        self.meta_dict = None

    def get_all_cell_ids(self):
        print(list(self.data.keys()))
        return list(self.data.keys())

    def extract_data(self, cell_id_list=None):
        if not cell_id_list:
            cell_id_list = self.cell_ids
        result_X1 = []
        result_X2 = []
        result_Y = []
        for cell_id in cell_id_list:
            X1_list, X2_list, Y_list, meta_dict = extract_eis_capacity_dataset(self.data, cell_id, self.soc_filter)
            result_X1.extend(X1_list)
            result_X2.extend(X2_list)
            result_Y.extend(Y_list)
        self.meta_dict = meta_dict
        return result_X1, result_X2, result_Y

    def get_meta_data(self):
        # 仅有一个电池的meta数据，包含cycle和soc
        return self.meta_dict


