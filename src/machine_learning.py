import pandas as pd
import datetime

from model.dataset import Dataset
from ga_solver_impl import GaSolverImpl

IS_ACTIVATE_CATEGORY = [\
    [True, True, False, False, False, True, False, False, False, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False],\
    [True, True, True, False, False, True, True, False, False, True, True, True, True, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, True, False, False, True, True, False, False],\
    [True, True, True, True, False, True, True, True, False, True, True, True, True, True, False, True, True, False, True, False, False, True, True, False, True, False, False, True, True, True, False, True, True, True, False],\
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],\
]

# IS_ACTIVATE_CATEGORY = [\
#     [True, True, False, False, False, True, False, False, False, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False],\
#     [True, True, True, False, False, True, True, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False],\
#     [False, False, True, False, False, True, False, False, False, True, False, True, True, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, True, False],\
#     [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],\
# ]

# IS_ACTIVATE_CATEGORY = [\
#     [True, True, False, False, False, True, False, False, False, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False],\
#     [True, True, True, False, False, True, True, False, False, True, True, True, True, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, True, False, False, True, True, False, False],\
#     [False, False, True, False, False, True, False, True, False, False, True, False, True, True, False, False, True, False, False, False, False, False, True, False, False, False, False, True, False, False, False, True, False, True, False],\
#     [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],\
# ]

def machine_learning(dataset:Dataset, speaker_count:int):
    if speaker_count < 1 or 4 < speaker_count:
        return
    
    selected = format_using_data(dataset = dataset, speaker_count = speaker_count)
    solver = GaSolverImpl(
        chromosome_length = selected.drop(columns= 'class').shape[1], 
        population_size = 100,
        pick_out_size = 10,
        individual_mutation_probability = 1.0,
        gene_mutation_probability = 0.03,
        iteration = 100,
        verbose = True
    )
    history = solver.solve(selected.drop(columns= 'class'), selected['class'])
    # print(solver.solve(selected.drop(columns= 'class'), selected['class']))

    history_df = pd.DataFrame(history).sort_values(["Max"], ascending=False)
    history_df.to_csv('out/history_{0}.csv'.format(datetime.date.today()))

def format_using_data(dataset:Dataset, speaker_count:int):
    drop_colmn = []
    drop_index = []
    
    # 削除するカラムの選定
    for i in range(len([attribute for attribute in dataset.attributes if attribute.type != 'class'])):
        if not IS_ACTIVATE_CATEGORY[speaker_count - 1][i // 6]:
            drop_colmn.append(dataset.attributes[i].name)
    
    # 削除するインデックスの選定
    for i, cell in enumerate(dataset.pandas['f{0}'.format(speaker_count * 6)]):
        if pd.isna(cell):
            drop_index.append(i)

    selected = dataset.pandas.drop(columns = drop_colmn, index = dataset.pandas.index[drop_index])
    selected = selected.fillna(0)

    return selected
