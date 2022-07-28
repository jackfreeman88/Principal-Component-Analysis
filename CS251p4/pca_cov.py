'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
Jack Freeman
CS 251 Data Analysis Visualization
Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA_COV:
	'''
	Perform and store principal component analysis results
	'''

	def __init__(self, data):
		''''''
		self.data = data

		# vars: Python list. len(vars) = num_selected_vars
		#	String variable names selected from the DataFrame to run PCA on.
		#	num_selected_vars <= num_vars
		self.vars = None

		# A: ndarray. shape=(num_samps, num_selected_vars)
		#	Matrix of data selected for PCA
		self.A = None

		# normalized: boolean.
		#	Whether data matrix (A) is normalized by self.pca
		self.normalized = None

		# A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
		#	Matrix of PCA projected data
		self.A_proj = None

		# e_vals: ndarray. shape=(num_pcs,)
		#	Full set of eigenvalues (ordered large-to-small)
		self.e_vals = None
		# e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
		#	Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
		self.e_vecs = None

		# prop_var: Python list. len(prop_var) = num_pcs
		#	Proportion variance accounted for by the PCs (ordered large-to-small)
		self.prop_var = None

		# cum_var: Python list. len(cum_var) = num_pcs
		#	Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
		self.cum_var = None

	def get_prop_var(self):
		'''(No changes should be needed)'''
		return self.prop_var

	def get_cum_var(self):
		'''(No changes should be needed)'''
		return self.cum_var

	def get_eigenvalues(self):
		'''(No changes should be needed)'''
		return self.e_vals

	def get_eigenvectors(self):
		'''(No changes should be needed)'''
		return self.e_vecs

	def covariance_matrix(self, data):
		'''Computes the covariance matrix of `data`'''
	   
		covmat = np.cov( data, rowvar=False)
		return covmat
	   

	def compute_prop_var(self, e_vals):
		'''Computes the proportion variance accounted for by the principal components (PCs).'''
		
		self.e_vals = e_vals
		eval_sum = np.sum(e_vals)
		proplist = []
		for i in e_vals:
			prop = i/eval_sum
			proplist.append(prop)
		return proplist 

	def compute_cum_var(self, prop_var):
		'''Computes the cumulative variance accounted for by the principal components (PCs).'''
		
		
		prop_var = self.compute_prop_var(self.e_vals)
		cum_vars = []
		a = 0
		for i in prop_var:
			a = a + i
			cum_vars.append(a)
		return cum_vars

	def pca(self, vars, normalize=False):
		'''Performs PCA on the data variables `vars` '''
		self.vars = vars
		self.A = self.data[self.vars]

		self.A = self.A.to_numpy()
		self.A_min = self.A.min()
		self.A_max = self.A.max()

		if normalize == True:
			print("Normalize")			
			self.A = (self.A - self.A_min)/(self.A_max-self.A_min) #normalize sep
			print(self.A.shape)
		
		covmatrix = self.covariance_matrix(self.A)
		self.e_vals, self.e_vecs = np.linalg.eig(covmatrix)
		self.prop_var = self.compute_prop_var(self.e_vals)
		self.cum_var = self.compute_cum_var(self.prop_var)

		self.normalized = normalize
		self.orig_means = np.mean(self.data)
		self.orig_scales = np.max(self.data) - np.min(self.data)

	def elbow_plot(self, num_pcs_to_keep=None):
		'''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
		x axis corresponds to top PCs included (large-to-small order)
		y axis corresponds to proportion variance accounted for '''
	   
		if num_pcs_to_keep is not None:
			x = range(1, num_pcs_to_keep + 1)
			y = self.cum_var[0: num_pcs_to_keep]
		else:
			x = range(1, len(self.cum_var) + 1)
			y = self.cum_var

		
		plt.xlabel('top PCs')
		plt.ylabel('(p) variance accounted for')
		plt.title('Cumulative variance accounted for')

		plt.plot(x, y, '.b-', linewidth=2, markersize=15)
		

	def pca_project(self, pcs_to_keep):
		'''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)'''
	  
		Ynew = self.A@ self.e_vecs
		self.A_proj = Ynew[:,pcs_to_keep]
		return self.A_proj
		

	def pca_then_project_back(self, top_k):
		'''Project the data into PCA space (on `top_k` PCs) then project it back to the data space'''
		
		pcs_to_keep = list(np.arange(top_k))
		self.pca_project(pcs_to_keep)
		
		A_r = (self.A_proj @ self.e_vecs[:,pcs_to_keep].T)		
		if (self.normalized):			
			A_r = A_r * (self.A_max-self.A_min) + self.A_min
		
		

		return A_r

		
