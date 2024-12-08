from helpers.formatDataset import formatDSEncoder
from helpers.formatDataset import applyLinearRegression3
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def predictSingleField(dataset, predicVar = "winner", predictColumns="Equipo",predictColumns2D=["Equipo"],uniquePredictValues = None, testSize = 0.2):
    le = LabelEncoder()
    dataset[predictColumns] = le.fit_transform(dataset[predictColumns])
    predictValuesEncoding = le.fit_transform(uniquePredictValues)
    dataset1 = formatDSEncoder(dataset)
    regressor, y_test, y_pred, sc = applyLinearRegression3(dataset1, predicVar=predicVar, predictColumns=predictColumns2D, testSize=testSize)
    predictions = {}
    for predictVarEncoding, originalPredictVar in zip(predictValuesEncoding, uniquePredictValues):
        dataPredict = pd.DataFrame({predictColumns: [predictVarEncoding]})
        prediction = regressor.predict(dataPredict)
        predictions[originalPredictVar] = prediction[0]
    #end for
    return predictions

#end def