using MLNetExamples.Console;

const float size = 107.3f;
var price = HousePricePrediction.Predict(size);
Console.WriteLine($"Predicted price for size: {size} m2 = {price * 100:C}k");

HousePricePredictionWithTrainedModel.TrainModel();
price = HousePricePredictionWithTrainedModel.Predict(size);
Console.WriteLine($"Predicted price for size: {size} m2 = {price * 100:C}k");
HousePricePredictionWithONNXModel.TrainModel();
HousePricePredictionWithONNXModel.Predict2(size);