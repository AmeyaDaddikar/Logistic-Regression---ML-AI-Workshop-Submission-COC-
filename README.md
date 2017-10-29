# Logistic-Regression---ML-AI-Workshop-Submission-COC-

##### Using [statsmodels.api](http://www.statsmodels.org), [module Logit](http://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html#statsmodels.discrete.discrete_model.Logit) (Sorta cheating)

This is my submission for the assignment given by the Community Of Coders (COC) on the topic of logistic regression.

Checkout my [other repository](https://github.com/AmeyaDaddikar/Logistic-Regression---ML-AI-Workshop-Submission-COC-/tree/fixed_alpha) to see my implementation of the cost function with fixed ALPHA and fixed EPOCHS.


# Important parameters that seem to be working well :
1. Accuracy      = 100.0 % 
2. Learning Rate = N.A.
3. EPOCHS        = N.A.

I had to use the API , because removing the NormalizeData function, the code began to throw RuntimeErrors and saw significant dips in accuracy. Folliwng is my Train function

'''python

	def train(X, y, model):

		m = len(y)
		logit_mod = sm.Logit(y, X)

		logit_res = logit_mod.fit(disp=0)
		#Saves the thetas in the JSON file
		model['theta'] = list(logit_res.params)
		return model
'''

I also took the liberty to merge the [Test File](data/test.csv) and the [Train File](data/test.csv) to create a [New training file](/data/combined_test.csv) which ensured that the model completely learns the expected behaviour.

The Flask web based application contains two pages viz [HomePage](app/templates/index.html) and [Result Page](app/templates/result.html)

![screenshot](https://raw.githubusercontent.com/AmeyaDaddikar/Logistic-Regression---ML-AI-Workshop-Submission-COC-/master/Documents/images_ws/screenshot.png)

In a nutshell, the homepage has a form wherein the user can put the various characteristics (input) of the phone, along with the expected price for re-selling. The result page gives the answer, as to whether the phone can be sold at the expectd price.

### Issues,drawbacks, and shortcomings

1. Since the data isn't normalized, I assume, even if the accuracy is 100%, the model might not work for other real life cases. I might be wrong on this though, and my broken IPhone 6 really didn't deserve 20k that Apple was providing xdxd.
2. This is more focused on learning to integrate flask rather than Logistic Regression. My other branch is more focused on the implementation of the Logit Regression.
3. The webpages are not responsive, i.e. they won't look good at all on devices with smaller screens.
4. The web-app sends and receives data in simplistic form. i.e. no fancy features like graphs are used.

