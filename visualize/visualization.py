import numpy as np

from config.config import VisConfig
import matplotlib as mpl
import matplotlib.pyplot as plt


class Visualization(object):
    def __init__(self):
        self.config = VisConfig()

    def plot_blast_out(self, result):
        mpl.rcParams.update({'font.size': 12})
        plt.rcParams['figure.figsize'] = [20, 25]

        if not result:
            return
        tbl_data = result.copy()
        del tbl_data['source']
        source_saml = result['source']
        values, tbl_data = self.get_align_rows(source_saml, tbl_data)
        n_rows = len(values)
        n_cols = len(source_saml)
        colors = []
        for row in values:
            cl = []
            for v in row:
                if v == ' ':
                    row_color = 'beige'
                elif v == '+':
                    row_color = 'goldenrod'
                elif v == '-':
                    row_color = 'lightsteelblue'
                else:
                    row_color = 'salmon'

                cl.append(row_color)
            colors.append(cl)
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        row_labels = [f"{key} | e:{tbl_data[key]['e_value']}" for key in tbl_data]
        columns = list(source_saml)
        table_ = ax.table(cellText=values,
                          cellColours=colors,
                          colLabels=columns,
                          rowLabels=row_labels,
                          colColours=['khaki' for _ in columns],
                          rowColours=['skyblue' for _ in list(tbl_data.keys())],
                          loc='center')
        cells = table_.properties()["celld"]
        for i in range(n_rows + 1):
            for j in range(n_cols):
                cells[i, j].set_text_props(ha="center")

        out_file = self.config.get_blast_out_path()
        plt.tight_layout()
        ax.autoscale(True)
        fig.savefig(str(out_file))

    @staticmethod
    def get_align_rows(source_str, data):
        actual_data = data.copy()
        rows = []
        for key in data:
            align_str = [' ' for _ in range(len(source_str))]
            record = data[key]
            query = record['query']
            match = record['match']
            x_left = source_str.find(query)
            align_str[x_left:x_left + len(match)] = list(match)
            align_str = align_str
            if len(align_str) > len(source_str):
                del actual_data[key]
                continue
            rows.append(align_str)
        return rows, actual_data
