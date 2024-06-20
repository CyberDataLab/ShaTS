# my_shap_package/tg_shap.py
import torch
import pandas as pd
import math
from .utils import all_subsets, all_subsets_without

def tg_shap(MODEL, SupportDataset, TestDataset, windowSize):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sumas = torch.zeros(len(TestDataset), device=device)
    shapley_values = torch.zeros(len(TestDataset), windowSize, device=device)

    subsets_total = all_subsets(windowSize)
    
    subsets_dict = {}
    coef_dict = {}
    respuestas = {}
    diccionario_shap_aciertos = {}
    
    for i in range(windowSize):
        subsets_dict[i] = all_subsets_without(windowSize, i)
        coef_dict[i] = math.factorial(i) * math.factorial(windowSize - i - 1) / math.factorial(windowSize)
    
    j = 0

    shap_columns = [f'shap{i+1}' for i in range(windowSize)]
    columns = ['idx', 'k', 'simulationRun', 'sample', 'clase_predicha', 'clase_real'] + shap_columns + ['prob_predicha', 'suma_shap', 'aciertos']
    
    df = pd.DataFrame(columns=columns)
    
    
    with torch.no_grad():
        for idx in range(len(TestDataset)):
            
            diferencias_medias_acumuladas = torch.zeros(windowSize, device=device)
            data = TestDataset[idx]
            pred_original = MODEL(data['given'].unsqueeze(0).to(device))
            class_original = torch.argmax(pred_original)
            prob_original = torch.softmax(pred_original, dim=1)[0][class_original]
            
            modified_data_batches = []
            subset_vecino_pairs = []            

            for subset in subsets_total:
                for vecino in range(len(SupportDataset)):
                    modified_data = TestDataset[idx]['given'].clone()
                    #print(vecino)
                    modified_data[subset] = SupportDataset[vecino]['given'][subset]
                    modified_data_batches.append(modified_data)
                    subset_vecino_pairs.append((tuple(subset), vecino))
            
    
            modified_data_batch = torch.stack(modified_data_batches).to(device)
            guesses = MODEL(modified_data_batch)
            probs = torch.softmax(guesses, dim=1)[:, class_original]
    
            for i, pair in enumerate(subset_vecino_pairs):
                #print(pair)
                respuestas[pair] = probs[i]
    
            pair_indices = {(subset, vecino): i for i, (subset, vecino) in enumerate(subset_vecino_pairs)}
    
            for instante in range(windowSize):
                subsets_con, subsets_sin = subsets_dict[instante]
    
                prob_con_media = torch.zeros(len(subsets_con)+1, device=device)
                prob_sin_media = torch.zeros(len(subsets_sin), device=device)
    
                for i, (s_con, s_sin) in enumerate(zip(subsets_con, subsets_sin)):
                    indices_con = [pair_indices[(tuple(s_con), vecino)] for vecino in range(len(SupportDataset))]
                    indices_sin = [pair_indices[(tuple(s_sin), vecino)] for vecino in range(len(SupportDataset))]
    
                    coef = coef_dict[len(s_sin)]
                    prob_con_media[i] = probs[indices_con].mean()*coef
                    prob_sin_media[i] = probs[indices_sin].mean()*coef
    
                diferencias = (prob_sin_media - prob_con_media)
                diferencias_medias_acumuladas[instante] += diferencias.sum()
    
            shapley_values[j] = diferencias_medias_acumuladas
    
            print(j)
            print("Suma de importancias:", diferencias_medias_acumuladas.sum())
            print(f"Probabilidad original: {torch.softmax(pred_original, dim=1)[0][class_original]}")
            sumas[j] = diferencias_medias_acumuladas.sum() - prob_original
            
            clase_predicha = torch.argmax(pred_original)
            clase_real = TestDataset[idx]['answer']
            
            if clase_predicha == clase_real:
                diccionario_shap_aciertos[j] = 1
            else:
                diccionario_shap_aciertos[j] = 0
            
            
            diferencias_medias_acumuladas = diferencias_medias_acumuladas.cpu().numpy()
            # rellenar dataframe
            
            df_data = {
                'idx': idx, 'k': len(SupportDataset), 'simulationRun': TestDataset[idx]['simulationRun'],
                'sample': TestDataset[idx]['sample'], 'clase_predicha': clase_predicha.cpu().numpy().item(),
                'clase_real': clase_real.cpu().numpy().item(), 'prob_predicha': prob_original.cpu().numpy().item(),
                'suma_shap': diferencias_medias_acumuladas.sum(), 'aciertos': diccionario_shap_aciertos[j]
            }
            for i in range(windowSize):
                df_data[f'shap{i+1}'] = diferencias_medias_acumuladas[i]
            
            df.loc[len(df)] = df_data
            
            
            j += 1
                
    return sumas, shapley_values, diccionario_shap_aciertos, df
