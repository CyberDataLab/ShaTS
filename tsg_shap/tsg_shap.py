import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from .utils import StrategySubsets, StrategyGrouping, StrategyPrediction, generateSubsets


class ShaTS:
    def __init__(self, 
                 model, 
                 supportDataset, 
                 strategySubsets=StrategySubsets.APPROX,
                 strategyGrouping=StrategyGrouping.TIME,
                 strategyPrediction=StrategyPrediction.MULTICLASS,
                 m = 5,
                 batchSize=32, 
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
        self.batchSize = batchSize
        self.verbose = verbose
        self.nclass = nclass
        self.classToExplain = classToExplain

        self._initializeGroups(nameFeatures, nameGroups, nameInstants)
        
        self.subsets_dict, self.allSubsets = generateSubsets(self.numGroups, self.m, self.strategySubsets)
        self.keysSupportSubsets = [(tuple(subset), entity) for subset in self.allSubsets for entity in range(len(self.supportDataset))]
        self.pairDicts = {(subset, entity): i for i, (subset, entity) in enumerate(self.keysSupportSubsets)}

        self.coefDict = self._generateCoefDict()
        self.meanPrediction = self._computeMeanPrediction()


    def _initializeGroups(self, nameFeatures, nameGroups, nameInstants):
        if self.strategyGrouping.value == StrategyGrouping.TIME.value:
            self.numGroups = self.windowSize
        elif self.strategyGrouping.value == StrategyGrouping.FEATURE.value:
            self.numGroups = self.numFeatures
        elif self.strategyGrouping.value == StrategyGrouping.MULTIFEATURE.value:
            if not self.customGroups:
                raise ValueError("Custom groups are required for MULTIFEATURE strategy.")
            self.numGroups = len(self.customGroups)
        
        self.nameFeatures = nameFeatures or [f'feature{i+1}' for i in range(self.numFeatures)]
        self.nameInstants = nameInstants or [f'instant{i+1}' for i in range(self.windowSize)]
        self.nameGroups = nameGroups or [f'group{i+1}' for i in range(self.numGroups)]

    def _generateCoefDict(self):
        coefDict = {}
        if self.strategySubsets.value == StrategySubsets.EXACT.value:
            for i in range(self.numGroups):
                coefDict[i] = math.factorial(i) * math.factorial(self.numGroups - i - 1) / math.factorial(self.numGroups)
        else:
            for i in range(self.numGroups):
                coefDict[i] = 1 / self.numGroups
        return coefDict

    def _computeMeanPrediction(self):
        meanPrediction = torch.zeros(self.nclass, device=self.device)
        with torch.no_grad():
            for i in range(0, len(self.supportDataset), self.batchSize):
                batch = self.supportDataset[i:i + self.batchSize]
                batch_tensor = torch.stack([data['given'] for data in batch]).to(self.device)
                meanPrediction += torch.sum(torch.softmax(self.model(batch_tensor), dim=1), dim=0) if self.strategyPrediction.value == StrategyPrediction.MULTICLASS.value else torch.sum(torch.sigmoid(self.model(batch_tensor)), dim=0)
        return meanPrediction / len(self.supportDataset)
    
    def _getPrediction(self, data):

        predOriginal = self.model(data['given'].unsqueeze(0).to(self.device))
        classOriginal = torch.argmax(predOriginal) if self.strategyPrediction.value == StrategyPrediction.MULTICLASS.value else 0
        if self.classToExplain != -1:
            classOriginal = self.classToExplain
        probOriginal = torch.softmax(predOriginal, dim=1)[0][classOriginal] if self.strategyPrediction.value == StrategyPrediction.MULTICLASS.value else torch.sigmoid(predOriginal)[0][0]

        return predOriginal, classOriginal, probOriginal
    
    def _modifyDataBatches(self, data):
        modifiedDataBatches = []
        if self.strategyGrouping.value == StrategyGrouping.TIME.value:
            for subset in self.allSubsets:
                dataTensor = data['given'].unsqueeze(0).expand(len(self.supportDataset), *data['given'].shape).clone().to(self.device)
                indexes = torch.tensor(list(subset), dtype=torch.long, device=self.device)
                dataTensor[:, indexes, :] = self.supportTensor[:, indexes, :].clone()
                modifiedDataBatches.append(dataTensor.clone())
        
        elif self.strategyGrouping.value == StrategyGrouping.FEATURE.value:
            for subset in self.allSubsets:
                dataTensor = data['given'].unsqueeze(0).expand(len(self.supportDataset), *data['given'].shape).clone().to(self.device)
                indexes = torch.tensor(list(subset), dtype=torch.long, device=self.device)
                for instant in range(self.windowSize):
                    dataTensor[:, instant, indexes] = self.supportTensor[:, instant, indexes].clone()
                modifiedDataBatches.append(dataTensor.clone())
        
        elif self.strategyGrouping.value == StrategyGrouping.MULTIFEATURE.value:

            for subset in self.allSubsets:
                dataTensor = data['given'].unsqueeze(0).expand(len(self.supportDataset), *data['given'].shape).clone().to(self.device)
                allIndexes = []
                
                for group in subset:
                    allIndexes.extend(self.customGroups[group])
                allIndexes = torch.tensor(allIndexes, dtype=torch.long, device=self.device)
                
                dataTensor[:, :, allIndexes] = self.supportTensor[:, :, allIndexes].clone()
                modifiedDataBatches.append(dataTensor)
            
        return modifiedDataBatches
    
    def _computeProbs(self, modifiedDataBatches, classOriginal):
        probs = []
        for i in range(0, len(modifiedDataBatches), self.batchSize):
            batch = torch.cat(modifiedDataBatches[i:i + self.batchSize]).to(self.device)
            guesses = self.model(batch)
            
            batchProbs = torch.softmax(guesses, dim=1)[:, classOriginal] if self.strategyPrediction.value == StrategyPrediction.MULTICLASS.value else torch.sigmoid(guesses)[:, 0]
            probs.extend(batchProbs.cpu())
            
        return torch.tensor(probs, device=self.device)

    def _computeDifferences(self, probs, instant, size):

        subsetsWith, subsetsWithout = self.subsets_dict[(instant, size)]
        probWith = torch.zeros(len(subsetsWith), device=self.device)
        probWithout = torch.zeros(len(subsetsWithout), device=self.device)

        for i, (sWith, sWithout) in enumerate(zip(subsetsWith, subsetsWithout)):
            indexesWith = [self.pairDicts[(tuple(sWith), entity)] for entity in range(len(self.supportDataset))]
            indexesWithout = [self.pairDicts[(tuple(sWithout), entity)] for entity in range(len(self.supportDataset))]
            coef = self.coefDict[len(sWithout)]
            probWith[i] = probs[indexesWith].mean() * coef 
            probWithout[i] = probs[indexesWithout].mean() * coef
            
        return probWith, probWithout

    
    def compute_tsgshap(self, testDataset):
        shatsValuesList = torch.zeros(len(testDataset), self.numGroups, device=self.device)

        with torch.no_grad():

            for idx in range(len(testDataset)):
                data = testDataset[idx]
                tsgshapvalues = torch.zeros(self.numGroups, device=self.device)

                predOriginal, classOriginal, probOriginal = self._getPrediction(data)

                modifiedDataBatches = self._modifyDataBatches(data)

                probs = self._computeProbs(modifiedDataBatches, classOriginal)

                for group in range(self.numGroups):
                    for size in range(self.numGroups):
                        probWith, probWithout = self._computeDifferences(probs, group, size)
                        tsgshapvalues[group] += (probWithout - probWith).mean()

                shatsValuesList[idx] = tsgshapvalues.clone()

                del modifiedDataBatches, probs, predOriginal, classOriginal, probOriginal, tsgshapvalues
                torch.cuda.empty_cache()
        
        return shatsValuesList
        
    def plot_tsgshap(self, 
                        shatsValues, 
                        testDataset = None, 
                        modelPredictions = None, 
                        path=None,
                        segmentSize=100):

            if modelPredictions is None:
                if testDataset is None:
                    raise ValueError("If modelPredictions is not provided, testDataset must be provided.")
                modelPredictions = [self._getPrediction(data) for data in testDataset]

            fontsize = 25
            size = shatsValues.shape[0]

            arr_plot = np.zeros((self.numGroups, size))
            arr_prob = np.zeros(size)

            for i in range(size):
                arr_plot[:, i] = shatsValues[i].cpu().numpy()
                arr_prob[i] = modelPredictions[i][2].detach().cpu().numpy()
            
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
                
                cbarAx = fig.add_axes([ax.get_position().x1 + 0.15,  
                                        ax.get_position().y0 - 0.05,          
                                        0.05,                          
                                        ax.get_position().height + 0.125])     

                cbar = fig.colorbar(cax, cax=cbarAx, orientation='vertical')
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
                    textName = 'TSG-SHAP (Temporal)'
                    nameColumns = self.nameInstants
                elif self.strategyGrouping.value == StrategyGrouping.FEATURE.value:
                    ylabel = 'Feature'
                    textName = 'TSG-SHAP (Feature)'
                    nameColumns = self.nameFeatures
                elif self.strategyGrouping.value == StrategyGrouping.MULTIFEATURE.value:
                    ylabel = 'MULTIFEATURE'
                    textName = 'TSG-SHAP ---MULTIFEATURE)'
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

