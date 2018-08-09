# Auto-FE(developing)
Automatically Feature Engineering for Machine Learning.

![Pokemon](https://github.com/AxsPlayer/auto-FE/blob/master/auto-FE/image/pokemon.png)

## Motivation.
In a lot of cases, there are bunch of dirty works to do with feature engineering, especially for traditional machine learning. 80 percent of time is wasted in feature engineering. Thus, it’s helpful if some normal and fixed steps in feature engineering could be done by code automatically.

## Introduction of test dataset.
The test dataset contains lots of Pokemons’ statistic number of their characters. Pokemon is one kind of magical animals living in anime of Japanese. The properties contain attack number, HP number, and so on so forth.

## Content of package.
The main content of this package is to adopt usual feature engineering steps for not only numerical but also categorical features.

- Numerical Features.
	- step1: Fill ‘NA’ with the Mean of vector.
	- step2: Translate all the values to positive numbers, if there is any negative number.
	- step3: Convert to normal distribution, if it’s currently not.
	- step4: Standardization the distribution.
	- step5: Round number to .3float.

- Categorical Features.
	- step1: Fill ’NA’ with ‘missing’ value.
	- step2: Combine categories whose ratio are under 0.01 into one 'Others' category.
	- step3: Encode category columns with One-hot Encoding method.
	- step4: Convert target column to numbers, if it’s not.

## Performance of Feature Engineering.
Measure the performance of feature engineering using accuracy of the same model with same parameters, before and after feature engineering.
- Task: Use all the other characters of Pokemon to predict whether it’s a Legendary one or not.
- Model used: Logistic Regression.(Sklearn)
- Measure Metrics: mean accuracy.
- Result: 
	- Before Numerical Feature Engineering: mean_accuracy=0.9675
	- After Numerical Feature Engineering: mean_accuracy=0.96875

From the comparasion of mean accuracy before and after transformation, it's obvious that transformation of numerical feature will increase accuracy.




