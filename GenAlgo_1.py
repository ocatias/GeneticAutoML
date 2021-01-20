from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import random
import multiprocessing
import time

import math
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.svm import NuSVC

from copy import deepcopy

class GenAutoML:

	max_selected_features_init = 50
	max_features_to_select = 300

	cid = 0
	nr_algos = 5

	individuals = []
	bestmodel = None

	algorithm_list = ["NuSVC", "RF", "Ridge", "XGBoost", "Radius"]
	#TODO update this at some point
	nr_numeric = {'NuSVC':0,'svc': 1, 'Radius' : 0, "XGBoost" : 0, "Ridge" : 0, "RF" : 0}

	#for example min sample split must be float < 1 or int > 1, check for similar problems!

	xgboost_params = {'colsample_bytree': list(np.linspace(0.01,1,100)), 'subsample' : list(np.linspace(0.01,1, 120, endpoint=False)), 'gamma': list(np.linspace(0, 1, 100)) + list(range(2,100,5)), 'min_child_weight' : list(range(1,100,2)), 'learning_rate' : list(np.linspace(0,1,500)), 'max_depth' : list(range(1,21)), 'n_estimators' : list(range(50, 3000, 50))}
	Ridge_params = {'alpha' : list(np.linspace(0.01,1,1000)) + list(np.linspace(1,250,500)), 'solver' : ["auto", "svd", "cholevsky", "lsqr", "sparse_cg", "sag", "saga"]}
	RF_params = {'min_samples_leaf' : list(range(1,20)),  'max_features' : ["auto", "sqrt", "log2"], 'min_samples_split': list(np.linspace(0.01, 0.9, 50)) + list(range(2,15)), 'n_estimators' : list(range(50, 5000, 50)), 'max_depth' : [None] + list(range(1, 40)), 'oob_score' : [True, False]}
	svc_params = {'degree' : list(range(0,2)), 'C' : list(np.linspace(0.01,0.99,30)) + list(range(1,50,2)), 'kernel' : ['linear', "poly", "rbf", "sigmoid"], 'gamma' : ["auto", "scale"] + list(np.linspace(0.001,0.99,50)) + list(range(1,5)), 'shrinking' : [True, False], 'class_weight' : [None, "balanced"]}
	Radius_params = {'algorithm': ["auto", "ball_tree", "kd_tree"], 'p': list(range(1,15)), 'Radius' : list(np.linspace(0.1,5,100)), 'weights' : ["uniform", "distance"]}
	NuSVC_params = {'nu':list(np.linspace(0.01,0.6,200)), 'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'degree':[1,2,3], 'gamma':['scale','auto'], 'shrinking':[True,False]}

	history_best_score = []
	history_avg_score = []
	history_worst_score = []

	def init_models(self, nr_features, necessary_number, algorithm, params):
		numeric = self.nr_numeric[algorithm]
		if self.genetic_feature_selection:
			self.individuals += [{'gene_length' : nr_features + numeric,  'start_index' : nr_features, 'algorithm' : algorithm,
		'gene':[None for i in range(nr_features + numeric)], 'score' : "TOTALLYNEWLOL", 'params' : self.generate_random_params(params)} for j in range(necessary_number)]
		else:
			self.individuals += [{'gene_length' : numeric,  'start_index' : 0, 'algorithm' : algorithm,
		'gene':[None for i in range(numeric)], 'score' : "TOTALLYNEWLOL", 'params' : self.generate_random_params(params)} for j in range(necessary_number)]


	def generate_random_params(self, params):
		p = {}
		for key in params:
			p[key] = random.choice(params[key])
		return p

	def fit(self, X, y):
		X_train, X_test, y_train, y_test = self.preprocessing(X,y)

		nr_features = X_train.columns.size
		necessary_number = math.floor(self.nr_individuals/self.nr_algos)

		self.init_models( nr_features, necessary_number, "Ridge" , self.Ridge_params)
		self.init_models( nr_features, necessary_number, "RF" , self.RF_params)
		self.init_models( nr_features, self.nr_individuals - (self.nr_algos-1)*necessary_number, "XGBoost" , self.xgboost_params)
		self.init_models( nr_features, necessary_number, "Radius" , self.Radius_params)
		self.init_models( nr_features, necessary_number, "NuSVC" , self.NuSVC_params)


		# self.init_models( nr_features, 0, "Ridge" , self.Ridge_params)
		# self.init_models( nr_features, 0, "RF" , self.RF_params)
		# self.init_models( nr_features, 0, "XGBoost" , self.xgboost_params)
		# self.init_models( nr_features, 0, "Radius" , self.Radius_params)
		# self.init_models( nr_features, self.nr_individuals, "NuSVC" , self.NuSVC_params)

		# self.init_models( nr_features, 0, "svc" , self.svc_params)

		print("Individuals:", len(self.individuals), "\nIterations:", self.nr_iterations, "\nThreads:", "all" if self.n_jobs == -1 else self.n_jobs, "\nElitism:", self.elitism, "\nCrossvalidate: ", (str(self.crossvalidation_folds) + " fold" if self.do_crossvalidate else "No"),
		"\nGenetic Feature Selection: ", "Yes" if self.genetic_feature_selection else "No", "\nEqual Distribution of individuals to algos: ", "Yes" if self.equal_distribution else "No")

		if self.genetic_feature_selection:
			print("\nWARNING GENETIC FEATURE SELECTION IS TURNED ON! This has a very high chance to drastically reduce the score on the test dataset.\n"*5)


		for i in range(0, self.nr_individuals):

			self.individuals[i]['id'] = self.individuals[i]['algorithm'] + str(self.cid)
			self.cid += 1

			#SELECT UP TO max_selected_features_init FEATURES
			features_to_select = []
			for j in range(self.individuals[i]['start_index']):
				if(random.randint(0,1) == 1 and len(features_to_select) < self.max_selected_features_init):
					random_feature = random.randint(0,self.individuals[i]['start_index'])
					while(random_feature is features_to_select):
						random_feature = random.randint(0,self.individuals[i]['start_index'])
					features_to_select.append(random_feature)

			for n in range(0, self.individuals[i]['gene_length']):
				if(n < self.individuals[i]['start_index']):
					if(n in features_to_select):
						self.individuals[i]['gene'][n] = 1
					else:
						self.individuals[i]['gene'][n] = 0
				else:
					self.individuals[i]['gene'][n] = random.random()


		if self.timelimit != None:
			timeout = time.time() + self.timelimit


		#MAIN LOOP
		for iterations in range(0, self.nr_iterations):


			if self.timelimit != None:
				if time.time() > timeout:
					print("Time limit reached")
					break

			print( '\n\nGeneration ', iterations)

			# print('Starting evaluation')
			#Evaluate all the individuals

			new_individuals = Parallel(n_jobs=self.n_jobs, verbose = self.verbose)(delayed(self.evaluate)(i,X_train, X_test, y_train, y_test) for i in range(self.nr_individuals))


			#Non parallelized loop
			# new_individuals = [self.evaluate(i,X_train, X_test, y_train, y_test) for i in range(self.nr_individuals)]

			self.individuals = deepcopy(new_individuals)
			# print(self.individuals)
			#Calculate information for update
			scores = []
			for indivual in self.individuals:
				# print(indivual['id'] + ": " + str(indivual['score']))
				scores.append(indivual['score'])

			curr_best_score = max(scores)
			curr_avg_score = sum(scores)/len(scores)
			curr_worst_score = min(scores)
			print('Best Score: ', curr_best_score)
			print('Average: ', curr_avg_score)
			print('Worst Score: ', curr_worst_score)

			self.history_best_score.append(curr_best_score)
			self.history_worst_score.append(curr_worst_score)
			self.history_avg_score.append(curr_avg_score)
			#Percentage chance of each indivual to get selected
			#Probability to be selected = score/total_score

			bestindividual = deepcopy(max(self.individuals, key=(lambda x: x['score'])))

			total_score = 0
			for individual in self.individuals:
				total_score += individual['score']
				if int(individual['score']) >= bestindividual['score']:
					bestindividual = deepcopy(individual)

			self.best_individual = deepcopy(bestindividual)
			print("Best individual score: ", bestindividual['score'])


			# self.bestmodel = bestindividual['model']
			# print(self.calc_error(self.bestmodel.predict(self.X, self.y),self.y))

			individuals_per_algo = {}
			percentages_per_algo = {}
			percentages = []
			distribution = {}
			best_score = {}


			for s in self.algorithm_list:
				indivuals_of_type_s = [x for x in self.individuals if x["algorithm"] == s]

				if(len(indivuals_of_type_s) == 0):
					self.history_distributions[s].append(0)
					self.history_best_scores[s].append(0)
					best_score[s] = 0
					continue
				individuals_per_algo[s] = indivuals_of_type_s
				distribution[s] = len(indivuals_of_type_s)
				best_score[s] = max(indivuals_of_type_s, key=(lambda x: x['score']))['score']

				self.history_distributions[s].append(len(indivuals_of_type_s))
				self.history_best_scores[s].append(best_score[s])

			print(distribution, "\n", best_score)

			for individual in self.individuals:
				percentages.append(individual['score']/total_score)

			for s in self.algorithm_list:
				tot = 0
				p = []
				if(s in individuals_per_algo):
					for i in individuals_per_algo[s]:
						tot += i['score']
					for i in individuals_per_algo[s]:
						p.append(i['score']/tot)

				percentages_per_algo[s] = p

			offsprings = []
			#Elitism: Keep the individual with the best scores
			if self.elitism:
				#Warning: LEADS TO ERRORS IF ONE OF THE MODELS NO INSTANCES
				for s in self.algorithm_list: #TODO

					dicts = [x for x in self.individuals if x["algorithm"] == s]
					# print(max(dicts, key=(lambda x: x['score'])))
					if len(dicts) > 0:
						offsprings.append(deepcopy(max(dicts, key=(lambda x: x['score']))))

			#Draw individuals that mate and generate their offspring
			while (len(offsprings) != self.nr_individuals):
			# for offspring_id in range(0, self.nr_individuals):
				#print("equal", self.equal_distribution)
				if self.equal_distribution:

					for alg in self.algorithm_list:
						if(alg in individuals_per_algo):
							for i in range(1,len(individuals_per_algo[alg])):
								#print(len(individuals_per_algo[alg]))
								individual1 = np.random.choice(a=individuals_per_algo[alg], p=percentages_per_algo[alg])
								individual2 = np.random.choice(a=individuals_per_algo[alg], p=percentages_per_algo[alg])
								offspring = self.create_offspring(individual1, individual2)
								offsprings.append(offspring)
				else:
					individual1 = np.random.choice(a=self.individuals, p=percentages)
					#pick second with same algorithm
					individual2 = np.random.choice(a=individuals_per_algo[individual1['algorithm']], p=percentages_per_algo[individual1['algorithm']])
					offspring = self.create_offspring(individual1, individual2)
					offsprings.append(offspring)

			self.individuals = offsprings


		self.bestmodel = self.gene_to_model(bestindividual)
		self.bestmodel.fit(self.gene_to_select_data(bestindividual['gene'], X_train),y_train)
		self.plot_history()


	def predict(self, X_in, y = None):
		predictions = self.bestmodel.predict(self.gene_to_select_data(self.best_individual['gene'], X_in))
		if y is not None:
			print("Final error on test data: ", self.calc_error(predictions, y))
		return predictions


	def __init__(self,n_jobs = -1, nr_individuals=100, nr_iterations=7, timelimit=None, elitism=True, mutation_chance_algo=0.3, mutation_chance_feature=0.1, equal_distribution=False, crossvalidate = False, cv = 5, genetic_feature_selection = False, verbose = 0, simple_data = False, max_nu = 0.6):
		print("Initialising class")
		#print(self.xgboost_params)
		self.NuSVC_params['nu'] = list(np.linspace(0.01,max_nu,200))

		self.cid = 0
		self.history_distributions = {"NuSVC":[], "RF":[], "Ridge":[], "XGBoost":[], "Radius":[]}
		self.history_best_scores = {"NuSVC":[], "RF":[], "Ridge":[], "XGBoost":[], "Radius":[]}
		self.history_best_score = []
		self.history_avg_score = []
		self.history_worst_score = []
		self.do_crossvalidate = crossvalidate
		self.crossvalidation_folds = cv
		self.genetic_feature_selection = genetic_feature_selection
		self.individuals = []
		self.verbose = verbose
		self.bestmodel = None

		# self.equal_distribution = False
		# self.nr_individuals = 100
		# self.nr_iterations = 7
		# self.timelimit = None
		# self.elitism = True
		# self.mutation_chance_feature = 0.01
		# self.mutation_chance_algo = 0.3

		#replace values
		self.equal_distribution = equal_distribution
		self.n_jobs = n_jobs
		self.nr_individuals = nr_individuals
		self.nr_iterations = nr_iterations
		self.elitism = elitism
		self.timelimit = timelimit
		self.mutation_chance_feature = mutation_chance_feature
		self.mutation_chance_algo = mutation_chance_algo
		self.simple_data = simple_data






	def return_best(self):
		return self.bestmodel

	def evaluate(self, id, X_train, X_test, y_train, y_test):
		individual = self.individuals[id]
		# print(id, ":\n", individual)
		if individual['score'] != "TOTALLYNEWLOL":
			return deepcopy(individual)

		model  = self.gene_to_model(individual)
		# print(X_train, y_train)
		# print(id, ":\n",individual)
		if self.do_crossvalidate:
			scores = cross_validate(model, self.gene_to_select_data(individual['gene'], self.X), self.y, cv=self.crossvalidation_folds, scoring="f1_macro")
			individual['score'] = min(scores['test_score'])
			if math.isnan(individual['score']):
				individual['score'] = 0

			# print(individual['score'])
		else:
			# print("fit")
			model.fit(self.gene_to_select_data(individual['gene'], X_train),y_train)
			# print("predict")
			predictions = model.predict(self.gene_to_select_data(individual['gene'], X_test))
			individual['score'] = self.calc_error(predictions, y_test)
		# individual['model'] = model
		return deepcopy(individual)

	def plot_history(self):
		self.plot_distributions()
		self.plot_best_score_per_algo()
		self.plot_scores()

	def plot_scores(self):
		plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
		plt.plot(range(len(self.history_best_score)), self.history_best_score, label='Best Score')
		plt.plot(range(len(self.history_best_score)), self.history_avg_score, label='Average Score')
		plt.plot(range(len(self.history_best_score)), self.history_worst_score, label='Worst Score')

		plt.xlabel('Generation')
		plt.ylabel('F1 Score')
		plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
		plt.title('Scores over Time')
		plt.show()


	def plot_best_score_per_algo(self):
		plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
		for s in self.algorithm_list:
			plt.plot(range(len(self.history_best_score)), self.history_best_scores[s], label=s)
		plt.xlabel('Generation')
		plt.ylabel('F1 Score')
		plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
		plt.title('Best Score per Algorithm over Time')
		plt.show()


	def plot_distributions(self):
		labels = [x for x in range(len(self.history_best_score))]
		width = 0.35       # the width of the bars: can also be len(x) sequence

		fig, ax = plt.subplots()
		ax.yaxis.set_major_locator(MaxNLocator(integer=True))
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		bottom = None
		for s in self.algorithm_list:
			if(bottom is None):
				ax.bar(labels, self.history_distributions[s], width, label=s)
				bottom = np.array(self.history_distributions[s])
			else:
				ax.bar(labels, self.history_distributions[s], width, label=s, bottom =bottom)
				bottom += np.array(self.history_distributions[s])

		ax.set_ylabel('Number of Individuals')
		ax.set_title('Algorithm Distribution over Time')
		ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
		plt.show()

	def calc_error(self, predictions, solutions):
		error = f1_score(solutions, predictions, average = 'macro')
		# print("Calculated error ", error)
		if math.isnan(error):
			# print("NaN found")
			error = 0
		return error

	#Combines two parents into a new individual,
	#also mutates the genes
	#Ideas: generalize for n-parents, different selection algorithms
	def create_offspring(self, individual1, individual2):
		offspring = deepcopy(dict(individual1)) #to keep the admistrative info
		offspring['id'] = offspring['algorithm'] + str(self.cid)
		self.cid += 1
		#print(individual1['id'] + " (" + str(individual1['score']) + ") + " + individual2['id'] + " (" + str(individual2['score']) + ") -> " + offspring['id'])
		offspring['score'] = 'TOTALLYNEWLOL'

		#mate numeric genes
		for i in range(0, individual1['gene_length']):
			if random.randint(0,1) == 1:
				offspring['gene'][i] = individual1['gene'][i]
			else:
				offspring['gene'][i] = individual2['gene'][i]

		#mate+mutate nominal genes
		for x in individual1["params"]:
			choice = np.random.choice([0,1])
			if choice==0:
				offspring['params'][x] = individual1['params'][x]
			elif choice==1:
				offspring['params'][x] = individual2['params'][x]


		#mutate numeric genes
		self.mutate(offspring)
		self.enforce_selected_feature_maximum(offspring)
		return offspring

	#Mutate a gene with a certain percentage
	def mutate(self, idv):
		#print(idv)
		for i in range(idv['gene_length']):
			if(i < idv['start_index']):
				if(np.random.choice(a=[0,1], p=[1-self.mutation_chance_feature, self.mutation_chance_feature]) == 1):
					idv['gene'][i] = random.randint(0,1)
			else:
				if(np.random.choice(a=[0,1], p=[1-self.mutation_chance_algo, self.mutation_chance_algo]) == 1):
					idv['gene'][i] = random.random()

		for x in idv["params"]:
			if random.random() < self.mutation_chance_algo:
				if (idv['algorithm'] == "Ridge"):
					idv['params'][x] = random.choice(self.Ridge_params[x])
				elif (idv['algorithm'] == "RF"):
					idv['params'][x] = random.choice(self.RF_params[x])
				elif (idv['algorithm'] == "svc"):
					idv['params'][x] = random.choice(self.svc_params[x])
				elif (idv['algorithm'] == "Radius"):
					idv['params'][x] = random.choice(self.Radius_params[x])
				elif (idv['algorithm'] == "NuSVC"):
					idv['params'][x] = random.choice(self.NuSVC_params[x])



	def gene_to_model(self, idv):
		gene = idv['gene']
		if (idv['algorithm'] == "XGBoost"):
			min_child_weight = idv['params']['min_child_weight']
			n_estimators = idv['params']['n_estimators']
			max_depth = idv['params']['max_depth']
			learning_rate = idv['params']['learning_rate']
			gamma = idv['params']['gamma']
			colsample_bytree = idv['params']['colsample_bytree']
			subsample = idv['params']['subsample']
			return XGBClassifier(subsample=subsample, n_estimators = n_estimators, min_child_weight=min_child_weight ,learning_rate = learning_rate, max_depth = max_depth,
			gamma = gamma, colsample_bytree = colsample_bytree,
								random_state=0)
		elif (idv['algorithm'] == "Ridge"):
			alpha = (idv['params'])['alpha']
			solver = (idv['params'])['solver']
			return RidgeClassifier(alpha=alpha)
		elif (idv['algorithm'] == 'RF'):
			n_estimators = 	(idv['params'])['n_estimators']
			max_depth = idv["params"]["max_depth"]
			max_features = idv["params"]["max_features"]
			min_samples_leaf = idv["params"]["min_samples_leaf"]
			min_samples_split = idv["params"]["min_samples_split"]
			oob_score = (idv['params'])['oob_score']
			return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
				min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score=oob_score, bootstrap=True, random_state=0)
		elif (idv['algorithm'] == "svc"):

			coef0 = gene[idv['start_index']]

			C = (idv["params"])["C"]
			degree = (idv["params"])["degree"]
			shrinking = (idv["params"])["shrinking"]
			kernel = (idv["params"])["kernel"]
			gamma = (idv["params"])["gamma"]
			return SVC(degree=degree, C=C, coef0=coef0, shrinking=shrinking, kernel=kernel, gamma=gamma, random_state=0)
		elif (idv['algorithm'] == "Radius"):
			alg = idv['params']['algorithm']
			Radius = idv["params"]["Radius"]
			p = idv["params"]["p"]
			weights = (idv['params'])['weights']
			return RadiusNeighborsClassifier(algorithm=alg, radius=Radius, p=p, weights=weights, outlier_label="most_frequent")
		elif (idv['algorithm'] == "NuSVC"):
			# print(idv["params"],"\n")
			nu = idv["params"]["nu"]
			kernel = idv["params"]["kernel"]
			degree = idv["params"]["degree"]
			gamma = idv["params"]["gamma"]
			shrinking = idv["params"]["shrinking"]

			if(kernel == 'poly' and degree > 1 and not self.simple_data):
				degree = 1
			return NuSVC(nu = nu, kernel = kernel, degree = degree, gamma = gamma, shrinking = shrinking, cache_size=20000, random_state=0)

	def count_nr_selected_feature(self, idv):
		selected_features = []
		# print(idv)
		for i in range(idv['start_index']):
			if(idv['gene'][i] == 1):
				selected_features.append(i)
		return(len(selected_features))

	def enforce_selected_feature_maximum(self, idv):
		selected_features = []
		for i in range(idv['start_index']):
			if(idv['gene'][i] == 1):
				selected_features.append(i)

		nr_features_to_remove = len(selected_features) - self.max_features_to_select
		features_to_remove = []
		if(nr_features_to_remove > 0):
			while(len(features_to_remove) < nr_features_to_remove):
				sel_feat = np.random.choice(a = selected_features)
				selected_features.remove(sel_feat)
				features_to_remove.append(sel_feat)

		for feature_idx in features_to_remove:
			idv['gene'][feature_idx] = 0;





	def gene_to_select_data(self, gene, data):
		if self.genetic_feature_selection:
			cols_to_drop = []

			drop_all_columns = True
			for i in range(0, len(gene)):
				if gene[i] == 0:
					cols_to_drop.append(i)
				else:
					drop_all_columns = False

			#Dropping all columns will lead to an calc_error
			#If this is the case we dont drop the first column
			if drop_all_columns:
				cols_to_drop = cols_to_drop.pop(0)

			return data.drop(data.columns[cols_to_drop], axis=1)
		else:
			return data

	def preprocessing(self, X,y):
		# X_pre = self.labelEncoding(X)
		self.X = X
		self.y = y
		return train_test_split(X, y, test_size = 0.3, random_state = 0)

	# def labelEncoding(self,data):
	# 	data_labelenc = data.copy()
	#
	# 	#Find columns with categorical values
	# 	s = (data_labelenc.dtypes == 'object')
	# 	object_cols = list(s[s].index)
	#
	# 	#Replace those by labels
	# 	for col in object_cols:
	# 		labels_df = pd.DataFrame(data_labelenc, columns=[col])
	# 		labelencoder = LabelEncoder()
	# 		data_labelenc[col] = labelencoder.fit_transform(labels_df[col])
	#
	# 	return data_labelenc
