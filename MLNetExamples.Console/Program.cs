using MLNetExamples.Console;

HousePricePrediction.Predict(1.4f);

HousePricePredictionWithTrainedModel.TrainModel();
HousePricePredictionWithTrainedModel.Predict(1.4f);
HousePricePredictionWithONNXModel.TrainModel();
HousePricePredictionWithONNXModel.Predict2(1.4f);