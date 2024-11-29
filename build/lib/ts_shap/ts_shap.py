# my_shap_package/ts_shap.py
import torch
import pandas as pd
import math
from .utils import all_subsets, all_subsets_without, generateRandomSubsets

def ts_shap(MODEL, SupportDataset, TestDataset, windowSize):
    
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



def approx_ts_shap(MODEL, SupportDataset, TestDataset, m, groupingCriteria='TIME', batch_size=32):
    
    windowSize = TestDataset[0]['given'].shape[0]
    num_predictores = TestDataset[0]['given'].shape[1]
    
    if groupingCriteria == 'TIME':
        nGroups = windowSize
    elif groupingCriteria == 'PREDICTOR':
        nGroups = num_predictores
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sumas = torch.zeros(len(TestDataset), device=device)
    shapley_values = torch.zeros(len(TestDataset), nGroups, device=device)
    
    subsets_total = set()
    
    subsets_dict = {}
    respuestas = {}
    diccionario_shap_aciertos = {}
    

    subsets_dict, subsets_total = generateRandomSubsets(nGroups, m)

    ###########################################################
    j = 0

    shap_columns = [f'shap{i+1}' for i in range(nGroups)]
    columns = ['idx', 'k', 'clase_predicha', 'clase_real'] + shap_columns + ['prob_predicha', 'suma_shap', 'aciertos']
    
    df = pd.DataFrame(columns=columns)
    
    with torch.no_grad():
        
        prediccion_media = torch.zeros(21, device=device)
        
        for i in range(0, len(SupportDataset), batch_size):
            batch = SupportDataset[i:i + batch_size]
            batch_tensor = torch.stack([data['given'] for data in batch]).to(device)
            prediccion_media += torch.sum(torch.softmax(MODEL(batch_tensor), dim=1), dim=0)
        
        prediccion_media /= len(SupportDataset)
        
        for idx in range(len(TestDataset)):
            data = TestDataset[idx]
                
            
            support_tensor = torch.stack([data['given'] for data in SupportDataset]).to(device)
            data_tensor = data['given'].unsqueeze(0).expand(len(SupportDataset), *data['given'].shape).clone().to(device)
            
            diferencias_medias_acumuladas = torch.zeros(nGroups, device=device)
            pred_original = MODEL(data['given'].unsqueeze(0).to(device))
            class_original = torch.argmax(pred_original)
            prob_original = torch.softmax(pred_original, dim=1)[0][class_original]
            
            modified_data_batches = []
            subset_vecino_pairs = []
            
            if groupingCriteria == 'TIME':
                for subset in subsets_total:
                    data_tensor = data['given'].unsqueeze(0).expand(len(SupportDataset), *data['given'].shape).clone().to(device)
                    indices = torch.tensor(list(subset), dtype=torch.long, device=device)
                    data_tensor[:, indices, :] = support_tensor[:, indices, :].clone()
                    modified_data_batches.append(data_tensor.clone())
                    subset_vecino_pairs.extend([(tuple(subset), vecino) for vecino in range(len(SupportDataset))])
            
            elif groupingCriteria == 'PREDICTOR':
                for subset in subsets_total:
                    data_tensor = data['given'].unsqueeze(0).expand(len(SupportDataset), *data['given'].shape).clone().to(device)
                    indices = torch.tensor(list(subset), dtype=torch.long, device=device)
                    for instante in range(windowSize):
                        data_tensor[:, instante, indices] = support_tensor[:, instante, indices].clone()
                    modified_data_batches.append(data_tensor.clone())
                    subset_vecino_pairs.extend([(tuple(subset), vecino) for vecino in range(len(SupportDataset))])
            
            # Procesar los datos en lotes
            probs = []
            for i in range(0, len(modified_data_batches), batch_size):
                batch = torch.cat(modified_data_batches[i:i + batch_size]).to(device)
                guesses = MODEL(batch)
                batch_probs = torch.softmax(guesses, dim=1)[:, class_original]
                probs.extend(batch_probs.cpu())
            probs = torch.tensor(probs, device=device)
    
            respuestas = {pair: probs[i] for i, pair in enumerate(subset_vecino_pairs)}
            pair_indices = {(subset, vecino): i for i, (subset, vecino) in enumerate(subset_vecino_pairs)}
            
            for instante in range(nGroups):
                for size in range(nGroups):
                    subsets_con, subsets_sin = subsets_dict[(instante, size)]
    
                    prob_con_media = torch.zeros(len(subsets_con), device=device)
                    prob_sin_media = torch.zeros(len(subsets_sin), device=device)
                        
                    for i, (s_con, s_sin) in enumerate(zip(subsets_con, subsets_sin)):
                        
                        indices_con = [pair_indices[(tuple(s_con), vecino)] for vecino in range(len(SupportDataset))]
                        indices_sin = [pair_indices[(tuple(s_sin), vecino)] for vecino in range(len(SupportDataset))]
                        
                        coef = 1 / nGroups
                        pesos = torch.ones(len(SupportDataset), device=device)
                            
                        pesos /= torch.sum(pesos)                            
                        
                        #prob_con_media[i] no debe ser la media sino la suma multiplicando por distancias
                        prob_con_media[i] = torch.sum(probs[indices_con] * pesos) * coef
                        prob_sin_media[i] = torch.sum(probs[indices_sin] * pesos) * coef
                        
    
                    diferencias = (prob_sin_media - prob_con_media)
                    diferencias_medias_acumuladas[instante] += diferencias.mean()
    
            shapley_values[j] = diferencias_medias_acumuladas            
    
            print(j)
            print("Suma de importancias:", diferencias_medias_acumuladas.sum() + prediccion_media[class_original])
            print(f"Probabilidad original: {torch.softmax(pred_original, dim=1)[0][class_original]}")
            sumas[j] = diferencias_medias_acumuladas.sum() - prob_original
            
            clase_predicha = torch.argmax(pred_original)
            clase_real = TestDataset[idx]['answer']
            
            if clase_predicha == clase_real:
                diccionario_shap_aciertos[j] = 1
            else:
                diccionario_shap_aciertos[j] = 0
            
            diferencias_medias_acumuladas = diferencias_medias_acumuladas.cpu().numpy()
            
            df_data = {
                'idx': idx, 'k': len(SupportDataset),
                'clase_predicha': clase_predicha.cpu().numpy().item(),
                'clase_real': clase_real.cpu().numpy().item(), 'prob_predicha': prob_original.cpu().numpy().item(),
                'suma_shap': diferencias_medias_acumuladas.sum(), 'aciertos': diccionario_shap_aciertos[j]
            }
            for i in range(nGroups):
                df_data[f'shap{i+1}'] = diferencias_medias_acumuladas[i]
            
            df.loc[len(df)] = df_data
            
            j += 1
            # Liberar memoria
            del support_tensor, data_tensor, modified_data_batches, guesses, probs
            torch.cuda.empty_cache()
                
    return sumas, shapley_values, diccionario_shap_aciertos, df


