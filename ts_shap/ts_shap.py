# my_shap_package/ts_shap.py
import torch
import pandas as pd
import math
from enum import Enum
from .utils import StrategySubsets, StrategyGrouping, StrategyPrediction, generate_subsets


class TSSHAP:
    def __init__(self, 
                 model, 
                 supportDataset, 
                 strategySubsets=StrategySubsets.APPROX_MK,
                 strategyGrouping=StrategyGrouping.TIME,
                 strategyPrediction=StrategyPrediction.MULTICLASS,
                 m = 5, 
                 batch_size=32, 
                 customGroups=None,
                 device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 nameFeatures = None,
                 nameGroups = None,
                 nameInstants = None,
                 verbose = 0,
                 nclass = 2,
                 ):
        
        print("He cambiado esto 3 veces")
        self.model = model
        self.supportDataset = supportDataset
        self.supportTensor = torch.stack([data['given'] for data in supportDataset]).to(device)
        self.windowSize = supportDataset[0]['given'].shape[0]
        self.numFeatures = supportDataset[0]['given'].shape[1]
        self.strategySubsets = strategySubsets
        self.strategyGrouping = strategyGrouping
        self.strategyPrediction = strategyPrediction
        self.customGroups = customGroups
        self.device = device
        self.m = m
        self.batch_size = batch_size
        self.verbose = verbose
        self.nclass = nclass

        self._initialize_groups(nameFeatures, nameGroups, nameInstants)
        
        #if verbose imprime lo que se está haciendo
        self.subsets_dict, self.all_subsets = generate_subsets(self.numGroups, self.m, self.strategySubsets)
        self.keys_support_subsets = [(tuple(subset), entity) for subset in self.all_subsets for entity in range(len(self.supportDataset))]
        self.pair_dicts = {(subset, entity): i for i, (subset, entity) in enumerate(self.keys_support_subsets)}

        self.coef_dict = self._generate_coef_dict()
        

        self.mean_prediction = self._compute_mean_prediction()


    def _initialize_groups(self, nameFeatures, nameGroups, nameInstants):
        if self.strategyGrouping == StrategyGrouping.TIME:
            self.numGroups = self.windowSize
        elif self.strategyGrouping == StrategyGrouping.FEATURE:
            self.numGroups = self.numFeatures
        elif self.strategyGrouping == StrategyGrouping.PROCESS:
            if not self.customGroups:
                raise ValueError("Custom groups are required for PROCESS strategy.")
            self.numGroups = len(self.customGroups)
        
        self.nameFeatures = nameFeatures or [f'feature{i+1}' for i in range(self.numFeatures)]
        self.nameInstants = nameInstants or [f'instant{i+1}' for i in range(self.windowSize)]
        self.nameGroups = nameGroups or [f'group{i+1}' for i in range(self.numGroups)]

    def _generate_coef_dict(self):
        coef_dict = {}
        if self.strategySubsets.value == StrategySubsets.EXACT.value:
            for i in range(self.numGroups):
                coef_dict[i] = math.factorial(i) * math.factorial(self.numGroups - i - 1) / math.factorial(self.numGroups)
        else:
            for i in range(self.numGroups):
                coef_dict[i] = 1 / self.numGroups
        return coef_dict

    def _compute_mean_prediction(self):
        mean_prediction = torch.zeros(self.nclass, device=self.device)
        with torch.no_grad():
            for i in range(0, len(self.supportDataset), self.batch_size):
                batch = self.supportDataset[i:i + self.batch_size]
                batch_tensor = torch.stack([data['given'] for data in batch]).to(self.device)
                mean_prediction += torch.sum(torch.softmax(self.model(batch_tensor), dim=1), dim=0)
        return mean_prediction / len(self.supportDataset)
    
    def _getPrediction(self, data):

        pred_original = self.model(data['given'].unsqueeze(0).to(self.device))
        class_original = torch.argmax(pred_original) if self.strategyPrediction.value == StrategyPrediction.MULTICLASS else 0
        prob_original = torch.softmax(pred_original, dim=1)[0][class_original] if self.strategyPrediction.value == StrategyPrediction.MULTICLASS else torch.sigmoid(pred_original)[0][0]

        return pred_original, class_original, prob_original
    
    def _modifyDataBatches(self, data):
        
        modified_data_batches = []
        if self.strategyGrouping.value == StrategyGrouping.TIME.value:
            for subset in self.all_subsets:
                data_tensor = data['given'].unsqueeze(0).expand(len(self.supportDataset), *data['given'].shape).clone().to(self.device)
                indexes = torch.tensor(list(subset), dtype=torch.long, device=self.device)
                data_tensor[:, indexes, :] = self.supportTensor[:, indexes, :].clone()
                modified_data_batches.append(data_tensor.clone())
        
        elif self.strategyGrouping.value == StrategyGrouping.FEATURE.value:
            for subset in self.all_subsets:
                data_tensor = data['given'].unsqueeze(0).expand(len(self.supportDataset), *data['given'].shape).clone().to(self.device)
                indexes = torch.tensor(list(subset), dtype=torch.long, device=self.device)
                for instant in range(self.windowSize):
                    data_tensor[:, instant, indexes] = self.supportTensor[:, instant, indexes].clone()
                modified_data_batches.append(data_tensor.clone())
        
        elif self.strategyGrouping.value == StrategyGrouping.PROCESS.value:
            for subset in self.all_subsets:
                data_tensor = data['given'].unsqueeze(0).expand(len(self.supportDataset), *data['given'].shape).clone().to(self.device)
                indexes = [self.customGroups[group] for group in subset]
                for instant in range(self.windowSize):
                    for group_indexes in indexes:
                        data_tensor[:, instant, group_indexes] = self.supportTensor[:, instant, group_indexes].clone()
                modified_data_batches.append(data_tensor.clone())

        return modified_data_batches
    
    def _computeProbs(self, modified_data_batches, class_original):
        probs = []
        for i in range(0, len(modified_data_batches), self.batch_size):
            batch = torch.cat(modified_data_batches[i:i + self.batch_size]).to(self.device)
            guesses = self.model(batch)
            
            batch_probs = torch.softmax(guesses, dim=1)[:, class_original] if self.strategyPrediction.value == StrategyPrediction.MULTICLASS else torch.sigmoid(guesses)[:, 0]
            probs.extend(batch_probs.cpu())
            
        return torch.tensor(probs, device=self.device)

    def _computeDifferences(self, probs, instant, size):

        subsets_with, subsets_without = self.subsets_dict[(instant, size)]
        prob_with = torch.zeros(len(subsets_with), device=self.device)
        prob_without = torch.zeros(len(subsets_without), device=self.device)

        for i, (s_with, s_without) in enumerate(zip(subsets_with, subsets_without)):
            indexes_with = [self.pair_dicts[(tuple(s_with), entity)] for entity in range(len(self.supportDataset))]
            indexes_without = [self.pair_dicts[(tuple(s_without), entity)] for entity in range(len(self.supportDataset))]
            coef = self.coef_dict[len(s_without)]
            prob_with[i] = probs[indexes_with].mean() * coef 
            prob_without[i] = probs[indexes_without].mean() * coef
        
        return prob_with, prob_without

    
    def compute_shap(self, testDataset):
        tsshapvalues_list = torch.zeros(len(testDataset), self.numGroups, device=self.device)

        with torch.no_grad():

            for idx in range(len(testDataset)):
                data = testDataset[idx]
                tsshapvalues = torch.zeros(self.numGroups, device=self.device)

                pred_original, class_original, prob_original = self._getPrediction(data)

                modified_data_batches = self._modifyDataBatches(data)

                probs = self._computeProbs(modified_data_batches, class_original)

                for group in range(self.numGroups):
                    for size in range(self.numGroups):
                        prob_with, prob_without = self._computeDifferences(probs, group, size)
                        tsshapvalues[group] += (prob_with - prob_without).mean()

                tsshapvalues_list[idx] = tsshapvalues.clone()
                
        return tsshapvalues_list






##################### LEGACY #####################



#FUNCION QUE SE USA ACTUALMENTE SIN EL PAQUETE
def USOaCTUAL(MODEL, SupportDataset, TestDataset, m, groupingCriteria='TIME', batch_size=32, multiGroups=None):
    
    windowSize = TestDataset[0]['given'].shape[0]
    num_predictores = TestDataset[0]['given'].shape[1]
    
    if groupingCriteria == 'TIME':
        nGroups = windowSize
    elif groupingCriteria == 'PREDICTOR':
        nGroups = num_predictores
    elif groupingCriteria == 'MULTIPREDICTOR':
        if multiGroups is None:
            raise ValueError("Para 'MULTIPREDICTOR', se debe proporcionar el argumento 'multiGroups'.")
        nGroups = len(multiGroups)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sumas = torch.zeros(len(TestDataset), device=device)
    shapley_values = torch.zeros(len(TestDataset), nGroups, device=device)
    
    subsets_total = set()
    
    subsets_dict = {}
    respuestas = {}
    diccionario_shap_aciertos = {}
    

    subsets_dict, subsets_total = tg_shap.generateRandomSubsets(nGroups, m)

    ###########################################################
    j = 0

    shap_columns = [f'shap{i+1}' for i in range(nGroups)]
    columns = ['idx', 'k', 'clase_predicha', 'clase_real'] + shap_columns + ['prob_predicha', 'suma_shap', 'aciertos']
    
    df = pd.DataFrame(columns=columns)
    
    with torch.no_grad():
        
        prediccion_media = torch.zeros(2, device=device)
        
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
            #class_original = 0
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
            
            elif groupingCriteria == 'MULTIPREDICTOR':
                for subset in subsets_total:
                    data_tensor = data['given'].unsqueeze(0).expand(len(SupportDataset), *data['given'].shape).clone().to(device)
                    indices = [multiGroups[group] for group in subset]
                    for instante in range(windowSize):
                        for group_indices in indices:
                            data_tensor[:, instante, group_indices] = support_tensor[:, instante, group_indices].clone()
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
    
            shapley_values[j] = diferencias_medias_acumuladas.clone()           
    
            print(j)
            print("Prediccion media:", prediccion_media[0], prediccion_media[1])
            print("Suma de importancias:", diferencias_medias_acumuladas.sum() + prediccion_media[class_original])
            print(f"Probabilidad original: {torch.softmax(pred_original, dim=1)[0][class_original]}")
            print(f"shapley_values: {diferencias_medias_acumuladas.sum()}")
            sumas[j] = diferencias_medias_acumuladas.sum() - prob_original
            
            clase_predicha = torch.argmax(pred_original)
            clase_real = TestDataset[idx]['answer']
            
            clase_real = torch.argmax(clase_real)
            
            #diccionario_shap_aciertos[j] = 1
            
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



#FUNCION POR ACTUALIZAR
def compute_ts_shap(MODEL, SupportDataset, TestDataset, m, groupingCriteria='TIME', batch_size=32):
    return 0
