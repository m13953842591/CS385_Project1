		/*
		 * 默认输入参数值
		 * C: regularization parameter
		 * tol: numerical tolerance
		 * max passes
		 */
		double C = 1; //对不在界内的惩罚因子
		double tol = 0.01;//容忍极限值
		int maxPasses = 5; //表示没有改变拉格朗日乘子的最多迭代次数
		
		/*
		 * 初始化a[], b, passes 
		 */
		
		double a[] = new double[x.length];//拉格朗日乘子
		this.a = a;
		
		//将乘子初始化为0
		for (int i = 0; i < x.length; i++) {
			a[i] = 0;
		}
		int passes = 0;
		
		
		while (passes < maxPasses) {
			//表示改变乘子的次数（基本上是成对改变的）
			int num_changed_alphas = 0;
			for (int i = 0; i < x.length; i++) {
				//表示特定阶段由a和b所决定的输出与真实yi的误差
				//参照公式(7)
				double Ei = getE(i);
				/*
				 * 把违背KKT条件的ai作为第一个
				 * 满足KKT条件的情况是：
				 * yi*f(i) >= 1 and alpha == 0 (正确分类)
				 * yi*f(i) == 1 and 0<alpha < C (在边界上的支持向量)
				 * yi*f(i) <= 1 and alpha == C (在边界之间)
				 * 
				 * 
				 * 
				 * ri = y[i] * Ei = y[i] * f(i) - y[i]^2 >= 0
				 * 如果ri < 0并且alpha < C 则违反了KKT条件
				 * 因为原本ri < 0 应该对应的是alpha = C
				 * 同理，ri > 0并且alpha > 0则违反了KKT条件
				 * 因为原本ri > 0对应的应该是alpha =0
				 */
				if ((y[i] * Ei < -tol && a[i] < C) ||
					(y[i] * Ei > tol && a[i] > 0)) 
				{
					/*
					 * ui*yi=1边界上的点 0 < a[i] < C
					 * 找MAX|E1 - E2|
					 */
					int j;
					/*
					 * boundAlpha表示x点处于边界上所对应的
					 * 拉格朗日乘子a的集合
					 */
					if (this.boundAlpha.size() > 0) {
						//参照公式(5)
						j = findMax(Ei, this.boundAlpha);
					} else 
						//如果边界上没有，就随便选一个j != i的aj
						j = RandomSelect(i);
					
					double Ej = getE(j);
					
					//保存当前的ai和aj
					double oldAi = a[i];
					double oldAj = a[j];
					
					/*
					 * 计算乘子的范围U, V
					 * 参考公式(4)
					 */
					double L, H;
					if (y[i] != y[j]) {
						L = Math.max(0, a[j] - a[i]);
						H = Math.min(C, C - a[i] + a[j]);
					} else {
						L = Math.max(0, a[i] + a[j] - C);
						H = Math.min(0, a[i] + a[j]);
					}
					
					
					/*
					 * 如果eta等于0或者大于0 则表明a最优值应该在L或者U上
					 */
					double eta = 2 * k(i, j) - k(i, i) - k(j, j);//公式(3)
					
					if (eta >= 0)
						continue;
					
					a[j] = a[j] - y[j] * (Ei - Ej)/ eta;//公式(2)
					if (0 < a[j] && a[j] < C)
						this.boundAlpha.add(j);
					
					if (a[j] < L) 
						a[j] = L;
					else if (a[j] > H) 
						a[j] = H;
					
					if (Math.abs(a[j] - oldAj) < 1e-5)
						continue;
					a[i] = a[i] + y[i] * y[j] * (oldAj - a[j]);
					if (0 < a[i] && a[i] < C)
						this.boundAlpha.add(i);
					
					/*
					 * 计算b1， b2
					 * 参照公式(6)
					 */
					double b1 = b - Ei - y[i] * (a[i] - oldAi) * k(i, i) - y[j] * (a[j] - oldAj) * k(i, j);
					double b2 = b - Ej - y[i] * (a[i] - oldAi) * k(i, j) - y[j] * (a[j] - oldAj) * k(j, j);
					
					if (0 < a[i] && a[i] < C)
						b = b1;
					else if (0 < a[j] && a[j] < C)
						b = b2;
					else 
						b = (b1 + b2) / 2;
					
					num_changed_alphas = num_changed_alphas + 1;
				}
			}
			if (num_changed_alphas == 0) {
				passes++;
			} else 
				passes = 0;
		}
		
		return new SVMModel(a, y, b);
