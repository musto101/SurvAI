from survshap import SurvivalModelExplainer, PredictSurvSHAP, ModelSurvSHAP

# create a class wrapper for the survshap classes
class shap():
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def fit(self):
        # create survival model explainer
        self.explainer = SurvivalModelExplainer(self.model, self.X_train, self.y_train)

        # create survival model shap
        self.model_shap = ModelSurvSHAP(self.explainer, self.model, self.X_train, self.y_train)

    def predict(self):
        # create survival model explainer
        self.predict_shap = PredictSurvSHAP(self.model_shap, self.X_test)

    def plot(self):
        # plot shap values
        self.predict_shap.plot()

    def summary_plot(self):
        # plot summary plot
        self.predict_shap.summary_plot()

    def force_plot(self, i):
        # plot force plot
        self.predict_shap.force_plot(i)

    def waterfall_plot(self, i):
        # plot waterfall plot
        self.predict_shap.waterfall_plot(i)

    def dependence_plot(self, i):
        # plot dependence plot
        self.predict_shap.dependence_plot(i)




