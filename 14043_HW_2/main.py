import linear_regression

print 'Enter dataset name:',
filename = raw_input()

if filename == 'seeds_dataset.txt' or filename == 'iris.data' or filename == 'AirQualityUCI.csv':
	print 'Enter initial delta:',
	initial_del = float(raw_input())
	X = linear_regression.parseFile(filename)
	linear_regression.f = filename
	theta = linear_regression.linear_regression(X, 1, 10000, initial_del)
	m, v = linear_regression.ten_fold_cross_validation(X, 10000, initial_del)
	print 'Theta:', theta
	print 'Mean of errors:', m
	print 'Standard deviation of errors:', v

else:
	print 'Kernel function to be implemented:'
	phi = 5
	while (phi > 4):
		print '1) Linear\n2) Polynomial\n3) Gausssian\n4) Polynomial and Gaussian(for simultaneous plots)\nEnter choice(1 - 4):',
		phi = input()
		if phi > 4:
			print 'Invalid option'
	print 'Enter initial delta:',
	initial_del = float(raw_input())
	
	X = linear_regression.parseFile(filename)
	linear_regression.f = filename
	theta = linear_regression.linear_regression(X, phi, 10000, initial_del)
	if (phi != 4):
		print 'Theta:', theta