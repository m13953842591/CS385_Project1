import numpy as np 
from matplotlib import pyplot as plt 
import random

def linear_kernel(xi, xj):
	# x1: [n1, p] x2: [n2, p]
	# return [n1, n2]
	return np.dot(xi, xj.T)

def rbf_kernel(A, B, epsilon=0.01):
	# A: [NA, p] B: [NB, p]
	# return [NA, NB]
	NA = A.shape[0]
	NB = B.shape[0]
	A_square = np.sum(np.power(A,2), axis=1, keepdims=True)
	B_square = np.sum(np.power(B,2), axis=1, keepdims=True)
	dsq = np.dot(A_square, np.ones((1, NB))) +\
		  np.dot(np.ones((NA, 1)), B_square.T) - 2*np.dot(A, B.T)
	return np.exp(-dsq / epsilon)

class SVM:

	def __init__(self, x, y, kernel = linear_kernel):
		self.x = x #[n, p]
		self.y = y #[n, ]
		self.a = np.zeros((self.x.shape[0], ), dtype=np.float32)
		self.b = 0
		self.K = kernel
		self.sv_idx = [] # 支持向量对应的下标，初始为空集

	def predict(self, x):
		# x [n2, p]
		if len(x.shape) == 1: # if single vector input
			x = x.reshape((1, x.shape[0]))
		# get support vector
		n1 = self.sv_idx.shape[0]
		sv_a = self.a[self.sv_idx].reshape((1, n1)) # [1, n1]
		sv_y = self.y[self.sv_idx].reshape((1, n1)) # [1, n1]
		sv_x = self.x[self.sv_idx] # [n1, p]
		pred = np.sum((sv_a * sv_y) * self.K(x, sv_x), axis=1) + self.b
		return pred # [n2, ]

	def getE(self, i):
		ui = self.predict(self.x[i])
		return ui - self.y[i]
	
	def findMax(self, Ei):
		sv_x = self.x[self.sv_idx] # [n1, p]
		sv_y = self.y[self.sv_idx] # [n1, ]
		u = self.predict(sv_x) # [n1, ]
		E = u - sv_y # [n1, ]
		idx = np.argmax(np.abs(E - Ei))
		return self.sv_idx[idx]
		
	def train(self, C=1, tol=0.01):
		"""
		利用SMO算法训练SVM
		:param C: 松弛变量惩戒因子
		:param tol: 容忍极限值
		"""
		n = self.x.shape[0]
		count = 5 
		while count > 0:
			flag = False
			for i in range(n):
				Ei = self.getE(i)
				if (self.y[i] * Ei < -tol and self.a[i] < C) \
				or (self.y[i] * Ei > tol and self.a[i] > 0) \
				or (self.y[i] * Ei == 0 and (self.a[i] == 0 or self.a[i] == C)):
					# 满足KKT条件的情况是：
					# yi*f(i) >= 1 and alpha == 0 (正确分类)
					# yi*f(i) == 1 and 0<alpha < C (在边界上的支持向量)
					# yi*f(i) <= 1 and alpha == C (在边界之间) 
					j = i
					if not self.sv_idx:
						# find j to max|Ei - Ej|
						j = self.findMax(Ei)
					else:
						while j == i:
							j = random.choice(list(range(n)))

					Ej = self.getE(j)
					ai_old, aj_old = self.a[i], self.a[j]

					# 求a2_new的取值范围
					L, H = 0, C
					if(self.y[i] != self.y[j]):
						L = max(0, aj_old - ai_old)
						H = min(C, C + aj_old - ai_old)
					else:
						L = max(0, ai_old + aj_old - C)
						H = min(C, ai_old + aj_old)

					# 计算kernel
					Kij = self.K(self.x[i], self.x[j])
					Kii = self.K(self.x[i], self.x[i])
					Kjj = self.K(self.x[j], self.x[j])

					
					eta =  Kii + Kjj - 2 * Kij
					
					if eta <= 0:
						# 如果eta等于0或者小于0 说明目标函数是非正定的
						# 则表明a最优值应该在L或者H上
						# 这样目标函数的最小值的位置在aj = L 或 aj = H
						# 带入目标函数中，相应的aj = L还是H就可以知道了
						# 参考https://www.cnblogs.com/jerrylead/archive/2011/03/18/1988419.html
						s = self.y[i] * self.y[j]
						f1 = self.y[i](Ei + self.b) - self.a[i] * Kii - s * self.a[j] * Kij
						f2 = self.y[j](Ej + self.b) - s * self.a[i] * Kij - self.a[j] * Kjj
						L1 = self.a[i] + s*(self.a[j] - L)
						H1 = self.a[i] + s*(self.a[j] - H)
						psiL = L1*f1 + L*f2 + 0.5*L1*L1*Kii + 0.5*L*L*Kjj + s*L*L1*Kij
						psiH = H1*f1 + H*f2 + 0.5*H1*H1*Kii + 0.5*H*H*Kjj + s*H*H1*Kij
						if psiH > psiL:
							self.a[j] = L
						else:
							self.a[j] = H
					else:
						# 通常情况下目标函数是正定的，也就是说，能够在直线约束方向上求得最小值，
						self.a[j] = self.a[j] - self.y[j] * (Ei - Ej) / eta

					# 加入支持向量
					if (self.a[j] > 0) and (self.a[j] < C) and (j not in self.sv_idx):
						self.sv_idx.append(j)

					# 最大值最小值剪辑
					self.a[j] = max(L, self.a[j])
					self.a[j] = min(H, self.a[j])

					# 如果几乎没有更新, 那么就认为达到了收敛状态
					if(abs(self.a[j] - aj_old) < 1e-5)
						continue 

					# 更新ai
					self.a[i] = self.a[i] + self.y[i] * self.y[j] * (aj_old - self.a[j])

					# 加入支持向量
					if (self.a[i] > 0) and (self.a[i] < C) and (i not in self.sv_idx):
						self.sv_idx.append(i)

					b1 = b - Ei - self.y[i] * (self.a[i] - ai_old) * Kii - self.y[j] * (self.a[j] - aj_old) * Kij
					b2 = b - Ej - self.y[i] * (self.a[i] - ai_old) * Kij - self.y[j] * (self.a[j] - aj_old) * Kjj
					if 0 < self.a[i] and self.a[i] < C:
						b = b1
					elif 0 < self.a[j] and self.a[j] < C:
						b = b2
					else:
						b = (b1 + b2) / 2
					flag = True

			if not flag:
				count -= 1
					


