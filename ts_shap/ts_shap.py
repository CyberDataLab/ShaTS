# my_shap_package/ts_shap.py
import torch
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
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
                 classToExplain = -1
                 ):
        
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
        self.classToExplain = classToExplain

        self._initialize_groups(nameFeatures, nameGroups, nameInstants)
        
        self.subsets_dict, self.all_subsets = generate_subsets(self.numGroups, self.m, self.strategySubsets)
        self.keys_support_subsets = [(tuple(subset), entity) for subset in self.all_subsets for entity in range(len(self.supportDataset))]
        self.pair_dicts = {(subset, entity): i for i, (subset, entity) in enumerate(self.keys_support_subsets)}

        self.coef_dict = self._generate_coef_dict()
        

        self.mean_prediction = self._compute_mean_prediction()


    def _initialize_groups(self, nameFeatures, nameGroups, nameInstants):
        if self.strategyGrouping.value == StrategyGrouping.TIME.value:
            self.numGroups = self.windowSize
        elif self.strategyGrouping.value == StrategyGrouping.FEATURE.value:
            self.numGroups = self.numFeatures
        elif self.strategyGrouping.value == StrategyGrouping.PROCESS.value:
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
                mean_prediction += torch.sum(torch.softmax(self.model(batch_tensor), dim=1), dim=0) if self.strategyPrediction.value == StrategyPrediction.MULTICLASS.value else torch.sum(torch.sigmoid(self.model(batch_tensor)), dim=0)
        return mean_prediction / len(self.supportDataset)
    
    def _getPrediction(self, data):

        pred_original = self.model(data['given'].unsqueeze(0).to(self.device))
        class_original = torch.argmax(pred_original) if self.strategyPrediction.value == StrategyPrediction.MULTICLASS.value else 0
        if self.classToExplain != -1:
            class_original = self.classToExplain
        prob_original = torch.softmax(pred_original, dim=1)[0][class_original] if self.strategyPrediction.value == StrategyPrediction.MULTICLASS.value else torch.sigmoid(pred_original)[0][0]

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
            
            batch_probs = torch.softmax(guesses, dim=1)[:, class_original] if self.strategyPrediction.value == StrategyPrediction.MULTICLASS.value else torch.sigmoid(guesses)[:, 0]
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

    
    def compute_tsshap(self, testDataset):
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
                        tsshapvalues[group] += (prob_without - prob_with).mean()

                tsshapvalues_list[idx] = tsshapvalues.clone()

                #free memory
                del modified_data_batches, probs, pred_original, class_original, prob_original, tsshapvalues
                torch.cuda.empty_cache()
        
        return tsshapvalues_list
        
    def visualize_tsshap(self, 
                        shapley_values, 
                        testDataset = None, 
                        model_predictions = None, 
                        path=None,
                        segmentSize=100):

            if model_predictions is None:
                if testDataset is None:
                    raise ValueError("If model_predictions is not provided, testDataset must be provided.")
                model_predictions = [self._getPrediction(data) for data in testDataset]

            fontsize = 25
            size = shapley_values.shape[0]

            arr_plot = np.zeros((self.numGroups, size))
            arr_prob = np.zeros(size)

            for i in range(size):
                arr_plot[:, i] = shapley_values[i].cpu().numpy()
                arr_prob[i] = model_predictions[i][2].detach().cpu().numpy()
            
            vmin, vmax = -0.5, 0.5
            cmap = plt.get_cmap('bwr')

            nSegments = (size + segmentSize - 1) // segmentSize
            fig, axs = plt.subplots(nSegments, 1, figsize=(15, 25 * (max(10, self.numGroups)/36) * nSegments)) #15, 25 predictor
            
            if nSegments == 1:
                axs = [axs]
            
            for n in range(nSegments):
                realEnd = min((n + 1) * segmentSize, size)
                if n == nSegments - 1:
                    realEnd = arr_plot.shape[1]
                    arr_plot = np.hstack((arr_plot, np.zeros((self.numGroups, segmentSize - (size % segmentSize)))))
                    arr_prob = np.hstack((arr_prob, -np.ones(segmentSize - (size % segmentSize))))
                    size = arr_plot.shape[1]
                
                init = n * segmentSize
                end = min((n + 1) * segmentSize, size)
                segment = arr_plot[:, init:end]
                ax = axs[n]


                ax.set_xlabel('Window', fontsize=fontsize)

                cax = ax.imshow(segment, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax, aspect='auto')
                
                cbar_ax = fig.add_axes([ax.get_position().x1 + 0.15,  
                                        ax.get_position().y0 - 0.05,          
                                        0.05,                          
                                        ax.get_position().height + 0.125])     

                cbar = fig.colorbar(cax, cax=cbar_ax, orientation='vertical')
                cbar.ax.tick_params(labelsize=fontsize)
                
                ax2 = ax.twinx()

                #prediction = arr_prob[init:end]

                prediction = arr_prob[init:realEnd]  # Ajustar a realEnd
                ax2.plot(np.arange(0, realEnd - init), prediction, linestyle='--', color='darkviolet', linewidth=4)
                    
                ax2.axhline(0.5, color='black', linewidth=1, linestyle='--')
                ax2.set_ylim(0, 1)
                ax2.tick_params(axis='y', labelsize=fontsize)

                ax2.set_ylabel('Model outcome', fontsize=fontsize)
                
                legend = ax2.legend(['Model outcome', 'Threshold'], fontsize=fontsize, loc = 'lower left', bbox_to_anchor=(0.0, -0.0))
                legend.get_frame().set_alpha(None)
                legend.get_frame().set_facecolor((0, 0, 0, 0))
                legend.get_frame().set_edgecolor('black')

                #switch case of the ylabel depending on the grouping strategy
                if self.strategyGrouping.value == StrategyGrouping.TIME.value:
                    ylabel = 'Time'
                    textName = 'TS-SHAP (Temporal)'
                    nameColumns = self.nameInstants
                elif self.strategyGrouping.value == StrategyGrouping.FEATURE.value:
                    ylabel = 'Feature'
                    textName = 'TS-SHAP (Feature)'
                    nameColumns = self.nameFeatures
                elif self.strategyGrouping.value == StrategyGrouping.PROCESS.value:
                    ylabel = 'Process'
                    textName = 'TS-SHAP (Process)'
                    nameColumns = self.nameGroups

                ax.set_ylabel(ylabel, fontsize=fontsize)
                ax.set_title(textName, fontsize=fontsize)

                ax.set_yticks(np.arange(self.numGroups))
                ax.set_yticklabels(nameColumns, fontsize=fontsize)
                
                xticks = np.arange(0, segment.shape[1], 5)  
                xlabels = np.arange(init, realEnd, 5)    

                xticks = xticks[:len(xlabels)]             

                ax.set_xticks(xticks)
                ax.set_xticklabels(xlabels, fontsize=fontsize)
        
            plt.tight_layout()

            if path is not None:
                plt.savefig(path)
            plt.show()
